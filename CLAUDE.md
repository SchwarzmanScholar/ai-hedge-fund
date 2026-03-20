# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An educational AI-powered multi-agent trading system that simulates a hedge fund. Uses LangGraph to orchestrate specialized AI agents (famous investor personalities + analytical agents) that analyze stocks and produce trading decisions.

## Commands

```bash
# Install dependencies
poetry install

# Copy and fill in environment variables
cp .env.example .env

# Run hedge fund analysis (CLI)
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

# Run with local LLMs via Ollama
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama

# Run backtester
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-12-31

# Run tests
pytest tests/

# Web app (FastAPI + React/Vite) - from repo root
cd app && npm run setup   # first time
./run.sh                  # Mac/Linux
run.bat                   # Windows
```

**Linting**: black (line length 420), isort (black profile), flake8

## Architecture

### Data Flow

```
Input (Tickers, Date Range)
  → Start Node
  → Parallel Analyst Agents (fetch financial data, generate signals)
  → Risk Management Agent (position limits, volatility-adjusted sizing)
  → Portfolio Manager Agent (final trading decisions)
  → Output (decisions with reasoning)
```

### Agent System

All agents are defined in [src/utils/analysts.py](src/utils/analysts.py) via `ANALYST_CONFIG`. There are 17 agents:
- **12 investor personality agents**: Warren Buffett, Charlie Munger, Ben Graham, Michael Burry, Bill Ackman, Cathie Wood, Phil Fisher, Peter Lynch, Stanley Druckenmiller, Mohnish Pabrai, Aswath Damodaran, Rakesh Jhunjhunwala
- **5 analytical agents**: Technical, Fundamentals, Growth, Sentiment, News Sentiment

Each analyst agent:
1. Fetches relevant financial data via [src/tools/api.py](src/tools/api.py) (FinancialDatasets.ai)
2. Constructs a prompt with that data
3. Calls an LLM and returns a signal (bullish/bearish/neutral) with confidence and reasoning

### Key Source Files

| File | Purpose |
|------|---------|
| [src/main.py](src/main.py) | CLI entry, builds and runs LangGraph workflow |
| [src/graph/state.py](src/graph/state.py) | `AgentState` TypedDict — all agents read/write this |
| [src/utils/analysts.py](src/utils/analysts.py) | Central registry of all analyst agents |
| [src/llm/models.py](src/llm/models.py) | LLM provider abstraction (12+ providers) |
| [src/utils/llm.py](src/utils/llm.py) | Helper: `call_llm()` with JSON mode and structured output |
| [src/tools/api.py](src/tools/api.py) | All financial data fetching (prices, financials, news, etc.) |
| [src/backtester.py](src/backtester.py) | CLI for backtesting |
| [src/backtesting/engine.py](src/backtesting/engine.py) | Backtesting loop and performance metrics |
| [app/backend/main.py](app/backend/main.py) | FastAPI backend for web UI |

### State Management

`AgentState` (in [src/graph/state.py](src/graph/state.py)) is the shared state dict passed through the LangGraph workflow. Key fields:
- `messages`: accumulated LangChain messages
- `data`: tickers, portfolio, dates, analyst signals
- `metadata`: model selection, show_reasoning flag

### LLM Providers

Supported via `ModelProvider` enum in [src/llm/models.py](src/llm/models.py): OpenAI, Anthropic, Groq, DeepSeek, Google, Ollama, xAI, GigaChat, Azure OpenAI, OpenRouter. Model lists are loaded from JSON config files. JSON mode support is detected per-model.

### Adding a New Analyst Agent

1. Create agent function in `src/agents/` following the pattern of existing agents
2. Register it in `ANALYST_CONFIG` in [src/utils/analysts.py](src/utils/analysts.py)
