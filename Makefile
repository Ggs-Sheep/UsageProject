make start:
	python3 -m venv /home/pi/UsageProject/venv
	source venv/bin/activate
	pip install -r requirements.txt
	python3 UsageReader.py