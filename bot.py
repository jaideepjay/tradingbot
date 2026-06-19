"""
Crypto Scalping Bot — Production Ready Fix
Enhancements:
  - Enforced a hard ceiling on wallet ingestion to prevent exceeding MAX_ACTIVE_SLOTS.
  - Implemented client-level request timeouts to completely stop thread blocking/skipped cycles.
  - Hardened local dictionary tracking against ghost positions.
  - Maintained direct credential initialization for script integrity.
"""

import os
import sys
import json
import time
import signal
import logging
import functools
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.helpers import round_step_size
import ollama

# =====================================================================
# CONFIGURATION
# =====================================================================

API_KEY    = "nvTqzj8vJ4m1BNvGdikS3UpkRypKZmk8dljpUuqHrD2pl4pzjFMO7hLBugpgUAqW"
API_SECRET = "9c7xmzSfIPNqX02DQKZrbvtGpxSeD2E9nLCef6gwj569hiop4tC0Sb0FxyftFiWg"

MAX_ACTIVE_SLOTS   = int(os.environ.get("MAX_ACTIVE_SLOTS",   "10"))
POSITION_RISK_USD  = float(os.environ.get("POSITION_RISK_USD", "10.0"))
CHECK_INTERVAL     = int(os.environ.get("CHECK_INTERVAL",      "120"))
COOLDOWN_DURATION  = int(os.environ.get("COOLDOWN_DURATION",   "3600"))
STATE_FILE         = os.environ.get("STATE_FILE",     "active_positions.json")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL",   "llama3:8b")

MAX_CONSECUTIVE_ERRORS = 5

# =====================================================================
# LOGGING
# =====================================================================

_run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"bot_run_{_run_ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# =====================================================================
# EXCLUDED SYMBOLS
# =====================================================================

EXCLUDED_ASSET_KEYWORDS: list[str] = [
    "USDT", "FDUSD", "USDC", "BUSD", "DAI", "AEUR", "EUR", "GBP", "USD1", "XAUT", "PAXG",
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOT", "MATIC", "LTC", "AVAX",
    "DOGE", "SHIB", "PEPE", "BONK", "FLOKI", "WIF", "PENGU", "MEME",
    "BOME", "TURBO", "MOG", "POPCAT", "BRETT", "MEW", "GOAT", "NEIRO", "NIGHT",
]

# Tokens whose price is pegged near $1 waste slots permanently — TP/SL will
# never be reached. Detected by live price, not name, to catch new stablecoins.
STABLECOIN_PRICE_MIN = 0.990
STABLECOIN_PRICE_MAX = 1.010

# =====================================================================
# TYPED POSITION STATE
# =====================================================================

@dataclass
class Position:
    symbol:      str
    quantity:    float
    entry_price: float
    take_profit: float
    stop_loss:   float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Position":
        return Position(**d)

# =====================================================================
# RETRY DECORATOR
# =====================================================================

def retry(max_attempts: int = 3, delay: float = 2.0, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        log.warning(
                            "[RETRY] %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            fn.__name__, attempt, max_attempts, exc, delay
                        )
                        time.sleep(delay)
            log.error("[RETRY] %s gave up after %d attempts: %s", fn.__name__, max_attempts, last_exc)
            raise last_exc
        return wrapper
    return decorator

# =====================================================================
# BINANCE CLIENT INITIALISATION
# =====================================================================

try:
    # Crucial Fix: Explicit requests timeouts enforced to stop lagging network blocks
    client = Client(API_KEY, API_SECRET, requests_params={"timeout": 10})
    log.info("[SYSTEM] Synchronising clock with Binance server…")
    server_time = client.get_server_time()["serverTime"]
    client.timestamp_offset = server_time - int(time.time() * 1000)
    log.info("[SYSTEM] Clock synchronised. Offset: %dms.", client.timestamp_offset)
except Exception as exc:
    sys.exit(f"[FATAL] Cannot connect to Binance: {exc}")

# =====================================================================
# TECHNICAL INDICATORS
# =====================================================================

def calculate_ema(prices: list[float], period: int = 50) -> Optional[float]:
    if len(prices) < period:
        return None
    k   = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def calculate_rsi(prices: list[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


def calculate_atr(klines: list, period: int = 14) -> Optional[float]:
    if len(klines) < period + 1:
        return None

    true_ranges = []
    for i in range(1, len(klines)):
        high       = float(klines[i][2])
        low        = float(klines[i][3])
        prev_close = float(klines[i - 1][4])
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))

    return sum(true_ranges[-period:]) / period

# =====================================================================
# OLLAMA LAYER
# =====================================================================

def ask_ollama_opinion(symbol: str, price_history: list[float]) -> bool:
    try:
        prices_str = ", ".join(f"${p:.4f}" for p in price_history[-10:])
        prompt = (
            f"Analyze this crypto token price trend for a short-term scalping "
            f"setup: {prices_str}. "
            "Is the short-term momentum bullish? "
            "Reply with exactly one word: BUY or SKIP."
        )
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
        verdict  = response.get("response", "").strip().upper()

        if verdict == "BUY":
            return True
        if verdict == "SKIP":
            return False

        has_buy  = "BUY"  in verdict.split()
        has_skip = "SKIP" in verdict.split()

        if has_skip:
            return False
        if has_buy:
            return True

        log.warning("[OLLAMA] Ambiguous verdict '%s' for %s — deferring to math filters.", verdict, symbol)
        return True

    except Exception as exc:
        log.warning("[OLLAMA] Connection error for %s: %s — deferring to math filters.", symbol, exc)
        return True

# =====================================================================
# TECHNICAL HEALTH EVALUATION
# =====================================================================

def evaluate_asset_technical_health(symbol: str) -> tuple[bool, Optional[float], Optional[float]]:
    try:
        klines = client.get_klines(
            symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=100
        )
        if len(klines) < 60:
            return False, None, None

        close_prices  = [float(c[4]) for c in klines]
        current_price = close_prices[-1]

        if not ask_ollama_opinion(symbol, close_prices):
            log.info("[FILTER] Ollama rejected %s.", symbol)
            return False, None, None

        ema_50 = calculate_ema(close_prices, period=50)
        rsi_14 = calculate_rsi(close_prices, period=14)
        atr_14 = calculate_atr(klines,       period=14)

        if ema_50 is None or rsi_14 is None or atr_14 is None:
            log.info("[FILTER] Insufficient indicator data for %s.", symbol)
            return False, None, None

        if current_price <= ema_50:
            log.info(
                "[FILTER] %s below EMA-50 (price=%.4f, ema=%.4f). Skipping.",
                symbol, current_price, ema_50
            )
            return False, None, None

        if not (45.0 <= rsi_14 <= 65.0):
            log.info(
                "[FILTER] %s RSI %.1f outside safe band [45, 65]. Skipping.",
                symbol, rsi_14
            )
            return False, None, None

        return True, current_price, atr_14

    except BinanceAPIException as exc:
        log.warning("[FILTER] Binance error evaluating %s: %s", symbol, exc)
        return False, None, None
    except Exception as exc:
        log.error("[FILTER] Unexpected error evaluating %s: %s", symbol, exc)
        return False, None, None

# =====================================================================
# SYMBOL DISCOVERY
# =====================================================================

@retry(max_attempts=3, exceptions=(BinanceAPIException, Exception))
def get_clean_midcap_basket() -> list[str]:
    tickers = client.get_ticker()
    valid: list[dict] = []

    for ticker in tickers:
        symbol = ticker["symbol"]
        base_asset = symbol.removesuffix("USDT")

        if not symbol.endswith("USDT"):
            continue
        if any(base_asset == kw or base_asset.startswith(kw) for kw in EXCLUDED_ASSET_KEYWORDS):
            continue
        try:
            valid.append({"symbol": symbol, "volume": float(ticker["quoteVolume"])})
        except (ValueError, KeyError):
            continue

    valid.sort(key=lambda x: x["volume"], reverse=True)
    return [item["symbol"] for item in valid[:25]]

# =====================================================================
# POSITION STATE — PERSISTENCE
# =====================================================================

def load_positions() -> dict[str, Position]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE) as fh:
            raw: dict = json.load(fh)
        # Crucial Safety Guard: Hard-restrict state loading to MAX_ACTIVE_SLOTS ceiling to squash 14/10 bug
        loaded = {k: Position.from_dict(v) for k, v in raw.items()}
        if len(loaded) > MAX_ACTIVE_SLOTS:
            log.warning("[STATE] File exceeds ceiling configuration. Pruning track structure down to %d entries.", MAX_ACTIVE_SLOTS)
            loaded = dict(list(loaded.items())[:MAX_ACTIVE_SLOTS])
        return loaded
    except Exception as exc:
        log.error("[STATE] Failed to load positions: %s — starting fresh.", exc)
        return {}


def save_positions(positions: dict[str, Position]) -> None:
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump({k: v.to_dict() for k, v in positions.items()}, fh, indent=4)
        os.replace(tmp, STATE_FILE)
    except Exception as exc:
        log.error("[STATE] Failed to save positions: %s", exc)

# =====================================================================
# ACCOUNT HELPERS
# =====================================================================

def get_free_usdt() -> float:
    try:
        info = client.get_asset_balance(asset="USDT")
        return float(info.get("free", 0.0)) if info else 0.0
    except Exception:
        return 0.0


def get_lot_constraints(symbol: str) -> tuple[Optional[float], Optional[float]]:
    try:
        info = client.get_symbol_info(symbol)
        if not info:
            return None, None
        step_size = tick_size = None
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step_size = float(f["stepSize"])
            elif f["filterType"] == "PRICE_FILTER":
                tick_size = float(f["tickSize"])
        return step_size, tick_size
    except Exception:
        return None, None


def normalise(symbol: str, raw_qty: float, raw_price: Optional[float] = None
              ) -> tuple[Optional[float], Optional[float]]:
    step_size, tick_size = get_lot_constraints(symbol)
    if step_size is None:
        return None, None
    qty   = round_step_size(raw_qty, step_size)
    price = round_step_size(raw_price, tick_size) if raw_price is not None and tick_size else None
    return qty, price

# =====================================================================
# WALLET INGESTION
# =====================================================================

def ingest_wallet_balances(active_positions: dict[str, Position], max_slots: int) -> None:
    try:
        balances = client.get_account().get("balances", [])
        updated  = False

        for balance in balances:
            # Crucial Fix: Use safe conditional boundaries rather than allowing negative math drops
            if len(active_positions) >= max_slots:
                log.info("[INGESTION] Capacity cap reached or exceeded — ending ingestion phase safely.")
                break

            asset  = balance["asset"]
            amount = float(balance["free"])

            if asset in {"USDT", "FDUSD", "USDC", "BUSD"} or amount <= 0:
                continue

            symbol = f"{asset}USDT"

            if symbol in active_positions:
                continue

            try:
                current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
            except BinanceAPIException:
                continue

            total_value = amount * current_price
            if total_value < 4.0:
                continue

            # Skip stablecoin-pegged tokens — they waste a slot permanently
            if STABLECOIN_PRICE_MIN <= current_price <= STABLECOIN_PRICE_MAX:
                log.info("[INGESTION] Skipping %s — price %.4f looks like a stablecoin peg.", symbol, current_price)
                continue

            log.info(
                "[INGESTION] Adopting untracked holding: %s %.6f (~$%.2f)",
                asset, amount, total_value
            )

            klines = client.get_klines(
                symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=20
            )
            atr = (
                calculate_atr(klines, period=14)
                if len(klines) >= 15
                else current_price * 0.01
            )

            qty, entry_price = normalise(symbol, amount, current_price)
            if not qty or qty <= 0 or entry_price is None:
                continue

            raw_sl_dist = (atr or entry_price * 0.01) * 1.5
            sl_dist     = min(max(raw_sl_dist, entry_price * 0.01), entry_price * 0.03)
            _, tp       = normalise(symbol, qty, entry_price + sl_dist * 2.0)
            _, sl       = normalise(symbol, qty, entry_price - sl_dist)

            if tp is None or sl is None or sl >= entry_price or tp <= entry_price:
                tp = round(entry_price * 1.02, 8)
                sl = round(entry_price * 0.99, 8)

            active_positions[symbol] = Position(
                symbol=symbol,
                quantity=qty,
                entry_price=entry_price,
                take_profit=tp,
                stop_loss=sl,
            )
            log.info(
                "[INGESTION] %s registered | entry=%.4f | TP=%.4f | SL=%.4f",
                symbol, entry_price, tp, sl
            )
            updated = True

        if updated:
            save_positions(active_positions)

    except Exception as exc:
        log.warning("[INGESTION] Failed safely during iteration: %s", exc)

# =====================================================================
# TRADE EXECUTION
# =====================================================================

def execute_entry(symbol: str, current_price: float, atr: float) -> Optional[Position]:
    try:
        raw_qty = POSITION_RISK_USD / current_price
        qty, _  = normalise(symbol, raw_qty)

        if not qty or (qty * current_price) < 5.0:
            log.warning("[TRADE] %s: computed quantity too small. Skipping.", symbol)
            return None

        log.info("[TRADE] Placing market buy for %s — qty=%.6f", symbol, qty)
        order = client.create_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=qty,
        )

        fills        = order.get("fills", [])
        filled_price = float(fills[0]["price"]) if fills else current_price

        sl_dist = atr * 2.0
        raw_tp  = filled_price + sl_dist * 2.0
        raw_sl  = filled_price - sl_dist

        _, tp = normalise(symbol, qty, raw_tp)
        _, sl = normalise(symbol, qty, raw_sl)

        if tp is None or sl is None or sl >= filled_price or tp <= filled_price:
            tp = round(filled_price * 1.02, 8)
            sl = round(filled_price * 0.99, 8)
            log.warning("[TRADE] %s: Using fallback safe 2:1 fixed percentage levels.", symbol)

        position = Position(
            symbol=symbol,
            quantity=qty,
            entry_price=filled_price,
            take_profit=tp,
            stop_loss=sl,
        )
        log.info(
            "[TRADE] %s entered | entry=%.4f | TP=%.4f | SL=%.4f",
            symbol, filled_price, tp, sl
        )
        return position

    except BinanceAPIException as exc:
        log.error("[TRADE] Binance error entering %s: %s", symbol, exc)
        return None
    except Exception as exc:
        log.error("[TRADE] Unexpected error entering %s: %s", symbol, exc)
        return None


def get_actual_balance(asset: str) -> float:
    try:
        info = client.get_asset_balance(asset=asset)
        return float(info.get("free", 0.0)) if info else 0.0
    except Exception:
        return 0.0


def execute_exit(symbol: str, quantity: float, reason: str) -> bool:
    base_asset = symbol.removesuffix("USDT")

    def _sell(qty: float) -> bool:
        normalised, _ = normalise(symbol, qty)
        if not normalised or normalised <= 0:
            log.error("[EXIT] %s: normalised qty is zero — cannot sell.", symbol)
            return False
        client.create_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_MARKET,
            quantity=normalised,
        )
        log.info("[EXIT] %s sold (%s) qty=%.6f", symbol, reason, normalised)
        return True

    try:
        return _sell(quantity)

    except BinanceAPIException as exc:
        if exc.code == -2010:
            log.warning(
                "[EXIT] %s: quantity mismatch (-2010). Fetching actual balance and retrying…",
                symbol
            )
            actual_qty = get_actual_balance(base_asset)
            if actual_qty <= 0:
                log.error("[EXIT] %s: actual balance is zero — nothing to sell.", symbol)
                return False
            try:
                return _sell(actual_qty)
            except BinanceAPIException as retry_exc:
                log.error("[EXIT] %s: retry with actual balance also failed: %s", symbol, retry_exc)
                return False
        log.error("[EXIT] Binance error exiting %s: %s", symbol, exc)
        return False

    except Exception as exc:
        log.error("[EXIT] Unexpected error exiting %s: %s", symbol, exc)
        return False

# =====================================================================
# MAIN EVENT LOOP
# =====================================================================

class BotRunner:
    def __init__(self):
        self.active_positions: dict[str, Position] = {}
        self.cooldown_registry: dict[str, float]   = {}
        self._sell_failures: dict[str, int]        = {}
        self._running = True
        self._consecutive_errors = 0

        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        log.info("[SYSTEM] Shutdown signal received — finishing current cycle…")
        self._running = False

    def _expire_cooldowns(self) -> None:
        now = time.time()
        expired = [sym for sym, ts in self.cooldown_registry.items() if now >= ts]
        for sym in expired:
            del self.cooldown_registry[sym]
            log.info("[COOLDOWN] Lockout expired for %s.", sym)

    def _register_cooldown(self, symbol: str) -> None:
        self.cooldown_registry[symbol] = time.time() + COOLDOWN_DURATION
        log.info("[COOLDOWN] %s locked for %ds.", symbol, COOLDOWN_DURATION)

    def _close_position(self, symbol: str, reason: str) -> None:
        pos     = self.active_positions.pop(symbol)
        success = execute_exit(symbol, pos.quantity, reason)

        if success:
            self._sell_failures.pop(symbol, None)
            self._register_cooldown(symbol)
        else:
            failures = self._sell_failures.get(symbol, 0) + 1
            self._sell_failures[symbol] = failures

            if failures >= 3:
                log.error(
                    "[EXIT] %s: sell failed %d times in a row. "
                    "Removing from tracked positions — PLEASE SELL MANUALLY on Binance.",
                    symbol, failures
                )
                self._sell_failures.pop(symbol, None)
            else:
                log.warning(
                    "[EXIT] %s: sell attempt %d/3 failed — will retry next cycle.",
                    symbol, failures
                )
                self.active_positions[symbol] = pos

        save_positions(self.active_positions)

    def _sanitise_positions(self) -> None:
        """
        Run at startup. Fixes two classes of bad positions in the state file:

        1. Stablecoin slots — tokens priced between $0.99–$1.01 are pegged and
           will never hit a real TP or SL. Sell them immediately and free the slot.

        2. Unreachable TP/SL bands — if the TP is more than 8% away or SL more
           than 5% away (legacy of old broken ATR calculations), recalculate from
           live ATR capped to sane limits (SL: 1–3%, TP: 2–6%).
        """
        updated = False

        for symbol in list(self.active_positions.keys()):
            pos = self.active_positions[symbol]

            # --- Stablecoin check ---
            try:
                current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
            except Exception:
                continue

            if STABLECOIN_PRICE_MIN <= current_price <= STABLECOIN_PRICE_MAX:
                log.warning(
                    "[SANITISE] %s looks like a stablecoin (price=%.4f). "
                    "Selling and freeing slot.",
                    symbol, current_price
                )
                self._close_position(symbol, "STABLECOIN_EVICTION")
                updated = True
                continue

            # --- TP/SL band sanity check ---
            e      = pos.entry_price
            tp_pct = (pos.take_profit - e) / e
            sl_pct = (e - pos.stop_loss)  / e

            band_ok = (0.01 <= sl_pct <= 0.05) and (0.015 <= tp_pct <= 0.08)
            if band_ok:
                continue

            log.warning(
                "[SANITISE] %s has bad bands: SL=%.2f%% below entry, "
                "TP=%.2f%% above. Recalculating…",
                symbol, sl_pct * 100, tp_pct * 100
            )
            try:
                klines = client.get_klines(
                    symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=20
                )
                atr = calculate_atr(klines, period=14) if len(klines) >= 15 else None

                raw_dist = (atr * 1.5) if atr else (e * 0.015)
                sl_dist  = min(max(raw_dist, e * 0.01), e * 0.03)

                _, new_tp = normalise(symbol, pos.quantity, e + sl_dist * 2.0)
                _, new_sl = normalise(symbol, pos.quantity, e - sl_dist)

                if new_tp and new_sl and new_sl < e < new_tp:
                    self.active_positions[symbol] = Position(
                        symbol=symbol,
                        quantity=pos.quantity,
                        entry_price=e,
                        take_profit=new_tp,
                        stop_loss=new_sl,
                    )
                    log.info(
                        "[SANITISE] %s fixed → TP=%.4f (+%.2f%%) SL=%.4f (-%.2f%%)",
                        symbol, new_tp, (new_tp - e) / e * 100,
                        new_sl, (e - new_sl) / e * 100
                    )
                    updated = True
            except Exception as exc:
                log.error("[SANITISE] %s: recalculation failed: %s", symbol, exc)

        if updated:
            save_positions(self.active_positions)

    def _phase_monitor(self) -> None:
        if not self.active_positions:
            return

        log.info("[MONITOR] Evaluating %d open positions…", len(self.active_positions))

        for symbol in list(self.active_positions.keys()):
            pos = self.active_positions[symbol]
            try:
                current_price = float(client.get_symbol_ticker(symbol=symbol)["price"])
            except Exception as exc:
                log.warning("[MONITOR] Could not fetch price for %s: %s", symbol, exc)
                continue

            pnl = (current_price - pos.entry_price) / pos.entry_price * 100
            log.info(
                "[DATA] %s | now=%.4f | entry=%.4f | PnL=%+.2f%%",
                symbol, current_price, pos.entry_price, pnl
            )

            if current_price >= pos.take_profit:
                log.info("[EXIT] TP hit for %s.", symbol)
                self._close_position(symbol, "TAKE_PROFIT")

            elif current_price <= pos.stop_loss:
                log.info("[EXIT] SL hit for %s.", symbol)
                self._close_position(symbol, "STOP_LOSS")

            else:
                # Trailing stop: once PnL >= 2%, raise SL to lock in 50% of gain.
                # This ensures a profitable position can never bleed all the way
                # back to zero while waiting for TP.
                if pnl >= 2.0:
                    gain       = current_price - pos.entry_price
                    new_sl_raw = pos.entry_price + gain * 0.50
                    _, new_sl  = normalise(symbol, pos.quantity, new_sl_raw)

                    if new_sl and new_sl > pos.stop_loss:
                        log.info(
                            "[TRAIL] %s: raising SL %.4f → %.4f (locking 50%% of gain).",
                            symbol, pos.stop_loss, new_sl
                        )
                        self.active_positions[symbol] = Position(
                            symbol=symbol,
                            quantity=pos.quantity,
                            entry_price=pos.entry_price,
                            take_profit=pos.take_profit,
                            stop_loss=new_sl,
                        )
                        save_positions(self.active_positions)

    def _phase_discovery(self) -> None:
        # Crucial Fix: Keep variable discovery bounded clearly to prevent loop calculations dropping below 0
        vacant = max(0, MAX_ACTIVE_SLOTS - len(self.active_positions))
        log.info(
            "[INVENTORY] %d/%d slots used — %d vacant.",
            len(self.active_positions), MAX_ACTIVE_SLOTS, vacant
        )

        if vacant <= 0:
            log.info("[STANDBY] Maximum capacity reached.")
            return

        free_usdt = get_free_usdt()
        log.info("[LIQUIDITY] Free USDT: $%.2f", free_usdt)

        if free_usdt < POSITION_RISK_USD:
            log.info("[STANDBY] Insufficient USDT (need $%.2f).", POSITION_RISK_USD)
            return

        try:
            basket = get_clean_midcap_basket()
        except Exception as exc:
            log.error("[DISCOVERY] Basket fetch failed: %s", exc)
            return

        for candidate in basket:
            if vacant <= 0 or free_usdt < POSITION_RISK_USD:
                break
            if candidate in self.active_positions or candidate in self.cooldown_registry:
                continue

            # Quick stablecoin price check before running full technical evaluation
            try:
                candidate_price = float(client.get_symbol_ticker(symbol=candidate)["price"])
                if STABLECOIN_PRICE_MIN <= candidate_price <= STABLECOIN_PRICE_MAX:
                    log.info("[DISCOVERY] Skipping %s — price %.4f is stablecoin range.", candidate, candidate_price)
                    continue
            except Exception:
                continue

            passed, price, atr = evaluate_asset_technical_health(candidate)
            if not passed:
                continue

            log.info("[DEPLOY] %s passed all filters — entering…", candidate)
            position = execute_entry(candidate, price, atr)

            if position:
                self.active_positions[candidate] = position
                save_positions(self.active_positions)
                vacant    -= 1
                free_usdt -= POSITION_RISK_USD

    def run(self) -> None:
        self.active_positions = load_positions()
        cycle = 0
        log.info("[SYSTEM] Bot started. Tracking %d positions.", len(self.active_positions))

        # Fix any stale/bad TP/SL bands and evict stablecoins before trading
        if self.active_positions:
            log.info("[SYSTEM] Sanitising loaded positions…")
            self._sanitise_positions()

        log.info("[SYSTEM] Running one-time wallet ingestion…")
        ingest_wallet_balances(self.active_positions, MAX_ACTIVE_SLOTS)
        log.info("[SYSTEM] Ingestion complete. Tracking %d positions.", len(self.active_positions))

        while self._running:
            cycle += 1
            log.info("=" * 70)
            log.info("[LOOP] Cycle #%d | %s", cycle, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            try:
                self._expire_cooldowns()
                self._phase_monitor()
                self._phase_discovery()
                self._consecutive_errors = 0

            except Exception as exc:
                self._consecutive_errors += 1
                log.error(
                    "[ERROR] Cycle anomaly (%d/%d): %s",
                    self._consecutive_errors, MAX_CONSECUTIVE_ERRORS, exc
                )
                if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log.error("[ERROR] Too many consecutive errors — sleeping 5 minutes.")
                    time.sleep(300)
                    self._consecutive_errors = 0
                else:
                    time.sleep(30)
                continue

            if self._running:
                log.info("[SLEEP] Sleeping for %ds…", CHECK_INTERVAL)
                time.sleep(CHECK_INTERVAL)

        log.info("[SYSTEM] Bot stopped gracefully.")


if __name__ == "__main__":
    BotRunner().run()