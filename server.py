"""
Bybit Pump-Retest Signal Bot — таймфрейм 30 минут
===================================================

Логика: живые монеты (оборот $5M+ и рост +20%+ за 24ч) -> композитный
индекс (RSI + Stoch%K + ROSC + WPR + %Rank + MFI + MACD + JAP, по мотивам
индикатора "INTELLECT_city - index inside %") -> пик 92-99% -> откат
в зону 70-79% -> сигнал в Telegram.

Это НЕ финансовый совет и НЕ готовая прибыльная торговая система —
только уведомление о совпадении технических условий. Настоятельно
рекомендуется проверить логику бэктестом перед использованием с
реальными деньгами.

Запуск:
    export TELEGRAM_TOKEN="8288068435:AAFStJROdw89XGqeG0ZGOrStQIjRVU77fh0"
    export TELEGRAM_CHAT_ID="7495689566"
    pip install requests pandas numpy
    python bot.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =========================================================
# КОНФИГУРАЦИЯ
# =========================================================

def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


TELEGRAM_TOKEN = os.getenv("8288068435:AAFStJROdw89XGqeG0ZGOrStQIjRVU77fh0", "")
TELEGRAM_CHAT_ID = os.getenv("7495689566", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    sys.exit(
        "❌ Не заданы TELEGRAM_TOKEN / TELEGRAM_CHAT_ID.\n"
        "   Задай их как переменные окружения перед запуском."
    )

MIN_TURNOVER_USD = _float("MIN_TURNOVER_USD", 5_000_000)
MIN_CHANGE_24H_PCT = _float("MIN_CHANGE_24H_PCT", 20.0)

PEAK_MIN = _float("PEAK_MIN", 92.0)
PEAK_MAX = _float("PEAK_MAX", 99.0)
RETEST_LOW = _float("RETEST_LOW", 70.0)
RETEST_HIGH = _float("RETEST_HIGH", 79.0)

INTERVAL = os.getenv("INTERVAL", "30")          # 30-минутные свечи
PEAK_LOOKBACK = _int("PEAK_LOOKBACK", 11)         # 11 свечей × 30 мин = 5.5 ч окно поиска пика
CANDLES = _int("CANDLES", 250)                     # ~5.2 дня истории
POLL_MINUTES = _int("POLL_MINUTES", 10)              # новая свеча каждые 30 мин, проверяем каждые 10

MAX_WORKERS = _int("MAX_WORKERS", 10)
HTTP_RETRIES = _int("HTTP_RETRIES", 3)
HTTP_RETRY_DELAY = _int("HTTP_RETRY_DELAY", 5)

STATE_PATH = os.getenv("STATE_PATH", "state.json")
LOG_PATH = os.getenv("LOG_PATH", "bot.log")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
BASE_URL = "https://api.bybit.com"


# =========================================================
# ЛОГИРОВАНИЕ
# =========================================================

log = logging.getLogger("pump_retest_bot")
log.setLevel(logging.INFO)

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
log.addHandler(_console)

_file = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_file.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
log.addHandler(_file)


# =========================================================
# СОСТОЯНИЕ (переживает перезапуск)
# =========================================================

@dataclass
class BotState:
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_scan_at: Optional[str] = None
    cycles_completed: int = 0
    pairs_checked: int = 0
    pairs_total: int = 0
    signals_sent: int = 0
    last_update_id: int = 0
    last_error: Optional[str] = None
    already_alerted: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "BotState":
        state = cls()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for key, value in saved.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
            except Exception:
                log.exception("Не удалось загрузить состояние — стартуем заново")
        return state

    def save(self, path: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.__dict__, f, ensure_ascii=False, indent=2)
        except OSError:
            log.exception("Не удалось сохранить состояние")

    def was_recently_alerted(self, symbol: str) -> bool:
        return self.already_alerted.get(symbol, 100.0) <= RETEST_HIGH

    def mark_alerted(self, symbol: str, index_value: float) -> None:
        self.already_alerted[symbol] = index_value


state = BotState.load(STATE_PATH)


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        if r.status_code != 200:
            log.warning("Telegram sendMessage вернул %s: %s", r.status_code, r.text[:200])
    except requests.RequestException as exc:
        log.warning("Ошибка отправки в Telegram: %s", exc)


def build_status_report() -> str:
    started = datetime.fromisoformat(state.started_at)
    uptime = datetime.now() - started
    hours, rem = divmod(int(uptime.total_seconds()), 3600)
    minutes = rem // 60
    err_line = f"\n⚠️ Последняя ошибка: {state.last_error}" if state.last_error else ""

    return (
        f"✅ <b>Бот работает</b> (ТФ 30 минут)\n\n"
        f"⏱ Аптайм: {hours}ч {minutes}мин\n"
        f"🔄 Циклов: {state.cycles_completed}\n"
        f"📊 Пар проверено: {state.pairs_checked}/{state.pairs_total}\n"
        f"🚀 Сигналов отправлено: {state.signals_sent}\n"
        f"💹 Фильтры: оборот ${int(MIN_TURNOVER_USD)//1_000_000}M+ | рост +{MIN_CHANGE_24H_PCT}%+\n"
        f"🕐 Последнее сканирование: {state.last_scan_at or 'ещё не было'}"
        f"{err_line}"
    )


def listen_commands() -> None:
    """Long-polling для /status и /help — работает в отдельном потоке."""
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            r = requests.get(url, params={"offset": state.last_update_id + 1, "timeout": 20}, timeout=25)
            updates = r.json().get("result", [])

            for upd in updates:
                state.last_update_id = upd["update_id"]
                text = upd.get("message", {}).get("text", "")

                if text == "/status":
                    send_telegram(build_status_report())
                elif text == "/help":
                    send_telegram("ℹ️ Команды:\n/status — статистика и состояние бота")

            if updates:
                state.save(STATE_PATH)

        except requests.RequestException as exc:
            log.warning("Ошибка long-poll: %s", exc)
            time.sleep(5)


# =========================================================
# BYBIT API
# =========================================================

def _get_json(path: str, params: dict) -> Optional[dict]:
    url = f"{BASE_URL}{path}"
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code != 200:
                log.warning("%s -> HTTP %s (попытка %s/%s)", path, r.status_code, attempt, HTTP_RETRIES)
                time.sleep(HTTP_RETRY_DELAY)
                continue

            data = r.json()
            if data.get("retCode") != 0:
                log.warning("%s -> retCode=%s (попытка %s/%s)", path, data.get("retCode"), attempt, HTTP_RETRIES)
                time.sleep(HTTP_RETRY_DELAY)
                continue

            return data

        except requests.RequestException as exc:
            log.warning("%s -> исключение %s (попытка %s/%s)", path, exc, attempt, HTTP_RETRIES)
            time.sleep(HTTP_RETRY_DELAY)

    return None


def fetch_hot_symbols() -> list[dict]:
    """Один запрос ко всем тикерам, отфильтрованный по обороту и росту за 24ч."""
    data = _get_json("/v5/market/tickers", {"category": "linear"})
    if data is None:
        return []

    result = []
    for item in data["result"]["list"]:
        symbol = item.get("symbol", "")
        if not symbol.endswith("USDT"):
            continue

        prev_price = float(item.get("prevPrice24h", 0) or 0)
        last_price = float(item.get("lastPrice", 0) or 0)
        turnover = float(item.get("turnover24h", 0) or 0)

        if prev_price <= 0 or turnover < MIN_TURNOVER_USD:
            continue

        change_pct = round((last_price - prev_price) / prev_price * 100, 2)
        if change_pct >= MIN_CHANGE_24H_PCT:
            result.append({"symbol": symbol, "change_24h": change_pct, "turnover": turnover})

    return result


def fetch_klines(symbol: str) -> Optional[pd.DataFrame]:
    data = _get_json("/v5/market/kline", {
        "category": "linear", "symbol": symbol, "interval": INTERVAL, "limit": CANDLES
    })
    if data is None:
        return None

    raw = data["result"]["list"]
    if not raw:
        return None

    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df.iloc[::-1].reset_index(drop=True)


# =========================================================
# КОМПОЗИТНЫЙ ИНДЕКС
# =========================================================

def compute_composite(df: pd.DataFrame) -> Optional[pd.Series]:
    """RSI + Stoch%K + ROSC + WPR + %Rank + MFI + MACD(норм.) + JAP, усреднённые в 0-100."""
    try:
        close, high, low, volume, open_ = df["close"], df["high"], df["low"], df["volume"], df["open"]
        n = 14

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        # Stochastic %K и Williams %R
        low_n, high_n = low.rolling(n).min(), high.rolling(n).max()
        span = (high_n - low_n).replace(0, np.nan)
        stoch = 100 * (close - low_n) / span
        wpr = 100 + (100 * (close - high_n) / span)

        # Money Flow Index
        hlc3 = (high + low + close) / 3
        mf = hlc3 * volume
        pos_mf = mf.where(hlc3 > hlc3.shift(1), 0).rolling(n).sum()
        neg_mf = mf.where(hlc3 < hlc3.shift(1), 0).rolling(n).sum()
        mfi = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))

        # Linear Correlation Oscillator и Percent Rank
        idx_s = pd.Series(range(len(close)), index=close.index)
        rosc = ((close.rolling(n).corr(idx_s) + 1) / 2) * 100
        rank = close.rolling(100).apply(lambda w: pd.Series(w).rank(pct=True).iloc[-1] * 100, raw=False)

        # MACD, нормализованный в 0-100
        ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9).mean()
        diff = macd - sig
        dmax = diff.abs().rolling(100).max().replace(0, np.nan)
        macd_n = 50 + (diff / dmax) * 50

        # "Japan Trade" — доля тела свечи в диапазоне, со знаком направления
        body = (close - open_).abs()
        rng = (high - low).replace(0, np.nan)
        # clip защищает от аномальных/противоречивых OHLC (напр. high < close при сбое API)
        body_ratio = (body / rng).clip(lower=0, upper=1)
        jap = pd.Series(
            np.where(close > open_, body_ratio * 100, (1 - body_ratio) * 100),
            index=close.index,
        )

        composite = (rsi + stoch + rosc + wpr + rank + mfi + macd_n + jap) / 8
        return composite.clip(lower=0, upper=100)  # финальная страховка на 0-100

    except Exception:
        log.exception("Ошибка расчёта композитного индекса")
        return None


# =========================================================
# ЛОГИКА СИГНАЛА: пик -> откат
# =========================================================

def check_pair(item: dict) -> Optional[dict]:
    symbol = item["symbol"]
    df = fetch_klines(symbol)
    if df is None or len(df) < 150:
        return None

    index_series = compute_composite(df)
    if index_series is None:
        return None

    current = index_series.iloc[-1]
    if pd.isna(current):
        return None
    current = round(float(current), 1)
    price = float(df["close"].iloc[-1])

    if not (RETEST_LOW <= current <= RETEST_HIGH):
        return None

    lookback = index_series.iloc[-(PEAK_LOOKBACK + 1):-1]
    peak_mask = (lookback >= PEAK_MIN) & (lookback <= PEAK_MAX)
    if not peak_mask.any():
        return None

    peak_value = float(lookback[peak_mask].max())
    peak_position = lookback[peak_mask].index[-1]
    candles_ago = len(df) - 1 - peak_position

    if current >= peak_value:
        return None  # должен быть строго ниже пика — это и есть откат

    return {
        "symbol": symbol,
        "price": price,
        "idx_current": current,
        "peak_value": round(peak_value, 1),
        "candles_ago": int(candles_ago),
        "change_24h": f"+{item['change_24h']}%",
        "turnover": f"${round(item['turnover'] / 1_000_000, 1)}M",
    }


def format_signal(r: dict) -> str:
    return (
        f"🚀 <b>ПАМП СИГНАЛ — LONG (Откат)</b>\n\n"
        f"Монета: <b>{r['symbol']}</b>\n"
        f"Рост 24ч: <b>{r['change_24h']}</b>\n"
        f"Оборот 24ч: <b>{r['turnover']}</b>\n\n"
        f"Пик индекса: <b>{r['peak_value']}%</b>\n"
        f"Текущий индекс: <b>{r['idx_current']}%</b>\n"
        f"Пик был: <b>{r['candles_ago']} свечи назад</b>\n"
        f"Текущая цена: <b>{r['price']}</b>\n\n"
        f"⏱ Таймфрейм: 30 минут"
    )


# =========================================================
# ГЛАВНЫЙ ЦИКЛ
# =========================================================

def run_cycle() -> None:
    started = time.time()

    hot_pairs = fetch_hot_symbols()
    state.pairs_total = len(hot_pairs)
    log.info("Отобрано горячих пар: %d", len(hot_pairs))

    if not hot_pairs:
        log.warning("Пустой список пар — возможен сбой API, короткая пауза.")
        state.last_error = f"Пустой ответ tickers в {datetime.now():%H:%M:%S}"
        time.sleep(60)
        return

    checked = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(check_pair, item): item for item in hot_pairs}

        for future in as_completed(futures):
            checked += 1
            state.pairs_checked = checked

            try:
                result = future.result()
            except Exception:
                log.exception("Ошибка проверки пары %s", futures[future]["symbol"])
                continue

            if result is None:
                continue

            symbol = result["symbol"]
            if state.was_recently_alerted(symbol):
                continue

            send_telegram(format_signal(result))
            state.signals_sent += 1
            state.mark_alerted(symbol, result["idx_current"])
            log.info(
                "СИГНАЛ: %s (индекс %.1f%%, пик %.1f%%, %d свечей назад)",
                symbol, result["idx_current"], result["peak_value"], result["candles_ago"],
            )

    state.last_scan_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state.cycles_completed += 1
    state.last_error = None
    state.save(STATE_PATH)

    elapsed = round(time.time() - started, 1)
    log.info("Цикл завершён за %.1f сек. Пауза %d мин.", elapsed, POLL_MINUTES)


def main() -> None:
    log.info(
        "Запуск. Таймфрейм=%s, пик=%.0f-%.0f%%, откат=%.0f-%.0f%%",
        INTERVAL, PEAK_MIN, PEAK_MAX, RETEST_LOW, RETEST_HIGH,
    )

    threading.Thread(target=listen_commands, daemon=True).start()

    send_telegram(
        "🚀 <b>Бот запущен!</b> (ТФ 30 минут)\n\n"
        f"• Оборот ${int(MIN_TURNOVER_USD)//1_000_000}M+ И рост +{MIN_CHANGE_24H_PCT}% за 24ч\n"
        f"• Пик индекса {PEAK_MIN}-{PEAK_MAX}%\n"
        f"• Откат в зону {RETEST_LOW}-{RETEST_HIGH}%\n"
        f"• /status — статистика"
    )

    while True:
        try:
            run_cycle()
        except Exception as exc:
            log.exception("Необработанная ошибка в основном цикле")
            state.last_error = str(exc)
            state.save(STATE_PATH)

        time.sleep(POLL_MINUTES * 60)


if __name__ == "__main__":
    main()
