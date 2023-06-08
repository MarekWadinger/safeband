import paho.mqtt.client as mqtt

# MQTT host information
host = "mqtt.cloud.uiam.sk"  # replace with your host's IP or hostname
port = 1883  # default MQTT port
topic = "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/dynamic_limits"  # topic to subscribe to

# MQTT callback functions


def on_connect(client, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(topic)


def on_message(client, msg):
    print(msg.payload)
    print("Received message: " + msg.payload.decode())


# Create MQTT client instance
client = mqtt.Client()

# Assign callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT host
client.connect(host, port, 60)

# Start the MQTT loop
client.loop_forever()
