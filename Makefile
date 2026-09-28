PY := .venv/bin/python

.PHONY: setup data analysis excel dashboard benchmark-mock test security all

setup:            ## create venv and install pinned dependencies
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

data:             ## generate the synthetic dataset (CSV + SQLite)
	$(PY) src/generate_data.py --seed 42

analysis:         ## SQL, statistics, figures and reports/insights.md
	$(PY) src/analysis/run_all.py

excel:            ## Excel workbook + Power BI star schema
	$(PY) src/excel_report.py && $(PY) src/powerbi_export.py

dashboard:        ## interactive Streamlit dashboard
	$(PY) -m streamlit run dashboard/app.py

benchmark-mock:   ## LLM benchmark pipeline without Ollama (simulated rows, tagged MOCK/)
	$(PY) llm_benchmark/run_benchmark.py --models llama3.2 qwen2.5-coder:1.5b mistral --repeats 3 --mock --out llm_benchmark/results/results.csv
	$(PY) llm_benchmark/analyze_benchmark.py --results llm_benchmark/results/results.csv --out-dir llm_benchmark/results

test:             ## full test suite
	$(PY) -m pytest -q

security:         ## static analysis + dependency vulnerability audit
	$(PY) -m bandit -r src dashboard llm_benchmark -q -ll   # fail on medium+ severity; low findings reviewed
	$(PY) -m pip_audit -r requirements.txt

all: data analysis excel test
