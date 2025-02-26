# PixelAgent: Unleash Your AgentX Crew 🚀✨

Meet **PixelAgent**—the fastest, most flexible way to build **Agent** helpers that tackle tasks with style 🌟. Choose your model, stack your powers, and let these autonomous champs roll with *blazing speed* ⚡, *rock-solid persistence* 🛡️, and *multimodal magic* 📸. Built for real work, loved by tinkerers—PixelAgent’s your go-to for getting stuff done, easy and fun 🎉.

Coders, creators, pros—anyone can jump in and make AgentX shine 🌈.

---

## Why PixelAgent? 🤔💡

- ⚡ **Super Fast**: Powers kick in instantly—zero lag!  
- 🛡️ **Always On**: Persistent AgentX keep the ball rolling.  
- 📸 **See Everything**: Text, images, PDFs—they handle it all.  
- 🧩 **Your Way**: Pick OpenAI, Anthropic, or add your own Python powers.  
- 🏢 **Ready to Roll**: Scalable, reliable—built for the big stuff.

---

## Get Started with AgentX 🌱⚙️

Grab the goods:
```bash
pip install pixelagent
```

Spin up your first AgentX:
```python
from pixelagent.anthropic import AgentX

agentx = AgentX(
    name="HelperX",
    system_prompt="You’re a brilliant assistant ✨.",
    model="claude-3-5-haiku-latest",
    max_tokens=1024,
    reset=True
)

result = agentx.execute("What’s the capital of France?")
print(result)  # "Paris! Ready for more? 🇫🇷"
```

---

## Build Your Crew 🛠️🌟

### 1. WebX Scout 🌐🔍
Digs up web info fast:
```python
from duckduckgo_search import DDGS
from pixelagent.anthropic import AgentX, power

@power
def search_the_web(keywords: str, max_results: int) -> str:
    with DDGS() as ddgs:
        results = ddgs.news(keywords, max_results=max_results)
        return "\n".join([f"{i}. {r['title']} - {r['body']}" for i, r in enumerate(results, 1)])

agentx = AgentX(
    name="WebX",
    model="claude-3-5-sonnet-20241022",
    system_prompt="You’re a web info whiz 🔍.",
    powers=[search_the_web]
)

print(agentx.execute("What’s new in tech? 💻"))
```

### 2. StockX Guide 💸📊
Lights up financial stats:
```python
import yfinance as yf
from pixelagent.openai import AgentX, power

@power
def get_stock_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info

agentx = AgentX(
    name="StockX",
    system_prompt="You’re a finance helper 💰.",
    powers=[get_stock_info]
)

print(agentx.execute("What’s up with FDS stock? 📈"))
```

### 3. VisionX Star 👁️‍🗨️📷
Sees and explains images:
```python
from pixelagent.openai import AgentX

agentx = AgentX(
    name="VisionX",
    system_prompt="You’re an image guru 🎨.",
    model="gpt-4o-mini"
)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
print(agentx.execute("What’s in this pic? 📸", attachments=url))
```

---

## AgentX Terminal Magic 🎨⚡

Your AgentX crew doesn’t just work—it *glows*. Peek at the fancy terminal output with `AgentXDisplay`:

- **Your Command**:  
  ```
  ┌── Your Command ──┐
  │ What’s new in tech?  │
  └─── 🚀 Launched by You 🚀 ───┘
  ```

- **Power Surge**:  
  ```
  ┌── Power Surge: search_the_web ⚡ ──┐
  │ ┌────────────┬──────────────┐   │
  │ │ 🔧 Inputs  │ 🎯 Output    │   │
  │ ├────────────┼──────────────┤   │
  │ │ {          │ 1. "Tech up" │   │
  │ │   "keywords": "tech news"   │   │
  │ │ }          │              │   │
  │ └────────────┴──────────────┘   │
  └─── 🔥 Power Unleashed 🔥 ───┘
  ```

- **AgentX Output**:  
  ```
  ┌── AgentX Output ──┐
  │ Tech’s booming—check it!  │
  └─── ✨ Powered Up ✨ ───┘
  ```

*Working hard?* See it grind:  
```
⏳ AgentX Grinding: Processing your request...
```

---

## Boost Your Flow 🌈🔧

- 🛠️ **Custom Powers**: Add your Python tricks—AgentX makes ‘em fly.  
- 🌟 **Quick Starts**: WebX, StockX, VisionX—ready to tweak and go.  
- 🎮 **AgentX Playground**: Test live, try fun challenges—“Grab news in 0.7s ⚡”—and see stats like “99% uptime! 🔥”.  

---

## Big Wins, No Sweat 🌟💼

- ⚡ **Speedy Wins**: Tasks done in *seconds*—faster than fast.  
- 📈 **Scale Easy**: Run 10 AgentX or 100—smooth every time.  
- 💡 **Time Saved**: Less work, more results—built for real impact.  

*Example*: “WebX grabbed 10 articles in 2s ⚡. Think auto-updates, anytime.”

---

## Join the Fun 🤝🎈

Share your AgentX creations on X with `#PixelAgent`. “My StockX nailed FDS stats—0.5s! 💸” Cool ideas spread fast.

---

## Jump In & Play ▶️

```bash
pip install pixelagent
```

Docs coming soon—until then, build your AgentX and have fun. Questions? Hit us on GitHub.

**PixelAgent: Call up AgentX. Add your powers. Make it happen. 🌈⚡**