import tensorflow as tf
import pandas as pd
import psutil as ps
import vcgencmd
import plotly.express as px
import plotly.graph_objects as go
from threading import Thread
import time
import os

thread_running = True

def perf_processing(perf):
    perf_df = pd.DataFrame(perf)
    perf_df.to_csv("perf.csv",index=False)

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=perf["Time"], y=perf["CPU_Usage (fit)"],
                        mode='lines',
                        name='CPU Usage (fit)'))
    fig.add_trace(go.Scatter(x=perf["Time"], y=perf["Memory_Usage (fit)"],
                        mode='lines',
                        name='Memory Usage (fit)'))
    
    fig.add_trace(go.Scatter(x=perf["Time"], y=perf["Core Voltage (fit)"],
                        mode='lines',
                        name='Core Voltage (fit)'))

    fig.add_trace(go.Scatter(x=perf["Time"], y=perf["Temperature (fit)"],
                        mode='lines',
                        name='Temperature (fit)'))
    
    # Evaluate
    fig.add_trace(go.Scatter(x=perf["Time"], y=perf["CPU_Usage (evaluate)"],
                        mode='lines',
                        name='CPU Usage (evaluate)'))
    fig.add_trace(go.Scatter(x=perf["Time"], y=perf["Memory_Usage (evaluate)"],
                        mode='lines',
                        name='Memory Usage (evaluate)'))
    
    fig.add_trace(go.Scatter(x=perf["Time"], y=perf["Core Voltage (evaluate)"],
                        mode='lines',
                        name='Core Voltage (evaluate)'))

    fig.add_trace(go.Scatter(x=perf["Time"], y=perf["Temperature (evaluate)"],
                        mode='lines',
                        name='Temperature (evaluate)'))

    fig.update_layout(title='System performances while running applications',
                   xaxis_title='Time',
                   yaxis_title='Usage (%)')
                   

    fig.write_html("perf.html")
    print("Files built")
    fig.show()



def perf_reader():
    global thread_running
    print("Starting perf monitoring...")
    perf = {"Time (fit)": [], "CPU_Usage (fit)": [], "Memory_Usage (fit)": [], "Core Voltage (fit)": [], "Temperature (fit)": [],
            "Time (evaluate)": [], "CPU_Usage (evaluate)": [], "Memory_Usage (evaluate)": [], "Core Voltage (evaluate)": [], "Temperature (evaluate)": []}
    start_time = time.time()


    while thread_running:
        time.sleep(0.1)
        perf['Time (fit)'].append(time.time() - start_time)
        perf['CPU_Usage (fit)'].append(ps.cpu_percent())
        perf['Memory_Usage (fit)'].append(ps.virtual_memory().percent)
        perf['Core Voltage (fit)'].append(vcgencmd.measure_volts('core'))
        perf['Temperature (fit)'].append(vcgencmd.measure_temp())
    

    while thread_running:
        time.sleep(0.1)
        perf['Time (evaluate)'].append(time.time() - start_time)
        perf['CPU_Usage (evaluate)'].append(ps.cpu_percent())
        perf['Memory_Usage (evaluate)'].append(ps.virtual_memory().percent)
        perf['Core Voltage (evaluate)'].append(vcgencmd.measure_volts('core'))
        perf['Temperature (evaluate)'].append(vcgencmd.measure_temp())
    
    perf_processing(perf)


def runner():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1)
    perf_fit = False

    loss, acc = model.evaluate(x_test, y_test)
    perf_evaluate = False

    print("Saving the model")
    model.save('my_mnist_model.h5')


if __name__ == '__main__':
    perf_fit = True
    perf_evaluate = True
    t1 = Thread(target=perf_reader)
    t2 = Thread(target=runner)

    t1.start()
    t2.start()

    t2.join()  # interpreter will wait until your process get completed or terminated
    print('Monitoring finished')

