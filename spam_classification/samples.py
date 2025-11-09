from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Literal


Lang = Literal["zh", "en"]
Category = Literal["spam", "ham", "mixed", "random"]


def _placeholders() -> Dict[str, str]:
    return {
        "url": random.choice([
            "https://example.com/claim",
            "https://safe.example.org/verify",
            "https://promo.example.net/offer",
        ]),
        "phone": random.choice([
            "02-1234-5678",
            "0912-345-678",
            "(555) 123-4567",
        ]),
        "code": "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6)),
        "time": random.choice(["10:30", "3 PM", "下午 3 點", "19:00"]),
    }


def _render(template: str, ph: Dict[str, str]) -> str:
    out = template
    for k, v in ph.items():
        out = out.replace("{" + k + "}", v)
    return out


SPAM_TEMPLATES_ZH: List[str] = [
    "恭喜獲得萬元禮券！立即點擊 {url} 領取，限時優惠。",
    "【重要】您的帳戶即將停用，請於 {url} 驗證資料。",
    "限時贈送 iPhone！撥打 {phone} 立刻申請。",
    "您已獲選貸款資格，馬上申請：{url}。",
    "抽中大獎！使用代碼 {code} 於 {url} 兌換。",
]

HAM_TEMPLATES_ZH: List[str] = [
    "晚餐想吃什麼？我 7 點到。",
    "今天的會議改到 {time}，麻煩更新行程。",
    "麻煩把檔案寄給我，謝謝！",
    "這週末一起爬山嗎？",
    "到家記得跟我說一聲。",
]

SPAM_TEMPLATES_EN: List[str] = [
    "Congratulations! You won a $1000 gift card. Claim at {url} now.",
    "URGENT: Your account will be suspended. Verify at {url}.",
    "Limited offer! Free iPhone. Call {phone} to apply.",
    "You are selected for a loan. Apply now at {url}.",
    "Winner! Redeem with code {code} at {url}.",
]

HAM_TEMPLATES_EN: List[str] = [
    "Hi, are we still on for the meeting at {time}?",
    "Please send me the report by EOD.",
    "Don't forget to bring the documents tomorrow.",
    "Happy birthday! Let's celebrate this weekend.",
    "I'll call you in 10 minutes.",
]


@dataclass
class Sample:
    text: str
    expected_label: Literal["spam", "ham"]
    lang: Lang


def generate_message(lang: Lang = "zh", category: Category = "random") -> Sample:
    ph = _placeholders()
    if lang == "en":
        spam_list, ham_list = SPAM_TEMPLATES_EN, HAM_TEMPLATES_EN
    else:
        spam_list, ham_list = SPAM_TEMPLATES_ZH, HAM_TEMPLATES_ZH

    if category == "spam":
        tpl = random.choice(spam_list)
        return Sample(text=_render(tpl, ph), expected_label="spam", lang=lang)
    if category == "ham":
        tpl = random.choice(ham_list)
        return Sample(text=_render(tpl, ph), expected_label="ham", lang=lang)
    if category in ("mixed", "random"):
        # 50/50 預設機率
        is_spam = random.random() < 0.5
        tpl = random.choice(spam_list if is_spam else ham_list)
        return Sample(text=_render(tpl, ph), expected_label="spam" if is_spam else "ham", lang=lang)
    # fallback
    tpl = random.choice(spam_list + ham_list)
    label = "spam" if tpl in spam_list else "ham"
    return Sample(text=_render(tpl, ph), expected_label=label, lang=lang)


def generate_batch(
    n: int = 5,
    lang: Lang = "zh",
    category: Category = "mixed",
    spam_ratio: float = 0.5,
) -> List[Dict[str, str]]:
    """
    產生 n 則樣本訊息。
    - category=spam/ham：全部同類別
    - category=mixed：依 spam_ratio 控制 spam 比例（0~1）
    - category=random：50/50
    回傳為字典清單，包含 text/expected_label/lang。
    """
    samples: List[Sample] = []
    if category == "spam":
        samples = [generate_message(lang, "spam") for _ in range(n)]
    elif category == "ham":
        samples = [generate_message(lang, "ham") for _ in range(n)]
    elif category == "mixed":
        for _ in range(n):
            is_spam = random.random() < max(0.0, min(1.0, spam_ratio))
            samples.append(generate_message(lang, "spam" if is_spam else "ham"))
    else:  # random
        samples = [generate_message(lang, "random") for _ in range(n)]

    return [
        {"text": s.text, "expected_label": s.expected_label, "lang": s.lang} for s in samples
    ]


def _cli():
    parser = argparse.ArgumentParser(description="Generate common spam/ham messages (zh/en)")
    parser.add_argument("--lang", choices=["zh", "en"], default="zh")
    parser.add_argument("--category", choices=["spam", "ham", "mixed", "random"], default="mixed")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--spam-ratio", type=float, default=0.5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    batch = generate_batch(n=args.n, lang=args.lang, category=args.category, spam_ratio=args.spam_ratio)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(batch)} samples to {args.out}")
    else:
        print(json.dumps(batch, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()

