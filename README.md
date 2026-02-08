# BCN Assumption-Driven Forecasting AI System (AF-AI)

An AI forecasting system that combines deterministic modeling with intelligent analysis; auto-detects data frequencies (daily/weekly/monthly/quarterly/yearly), allows setting up confidence-based scenarios (±5-20%), modeling discrete events, and provides AI-powered strategic insights through Groq or Ollama, all with complete transparency into every calculation.

## Key Features

- Assumption-Driven Forecasting - Transparent forecasts with explicit growth rates, confidence levels, and scenario modeling
- Multi-Frequency Support - Auto-detects daily, weekly, monthly, quarterly, yearly, and generic period patterns
- Scenario Generation - Automatic upside/downside scenarios (High: ±5%, Medium: ±10%, Low: ±20% confidence spreads)
- Event Modeling - Simulate discrete impacts (product launches, pricing changes) with custom multipliers and decay periods
- Data Integration - Upload, join, and edit multiple data sources with Excel export
- AI-Powered Strategic Assistant - Chat with your forecasts using Groq (cloud) or Ollama (local), get insights, answer "what-if" questions, and build assumptions conversationally
- CAGR Analysis - Geometric growth calculations with period-accurate rate conversions

## How To Run

1. Make sure you have Python 3.8+ and Ollama (and the models) set up, and clone this repository on your local machine.
2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install the dependencies:
```bash
pip install -r requirements.txt
```
4. Run the application and use either the local or the cloud models:
```bash
streamlit run app.py
```

## Contributing

Contributions are welcome!

## License

Distributed under the MIT License. 
