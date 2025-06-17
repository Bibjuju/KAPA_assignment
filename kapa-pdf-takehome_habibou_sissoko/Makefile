default: help

help:
	@echo "make run"

run:
	streamlit run app/streamlit_app.py --logger.level=info

format:
	isort src app
	black src app