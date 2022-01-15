make start:
	python3 -m venv /home/pi/Mnist/UsageProject/venv
	source /home/pi/Mnist/UsageProject/venv/bin/activate
	pip install -r requirements.txt
	python3 UsageReader.py