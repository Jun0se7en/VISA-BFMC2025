import json
import time
import logging
from src.utils.messages.allMessages import Location
from src.utils.messages.messageHandlerSender import messageHandlerSender
from twisted.internet import protocol
from src.utils.messages.allMessages import (
    Position,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The server itself. Creates a new Protocol for each new connection and has the info for all of them.
class tcpClient(protocol.ClientFactory):
    def __init__(self, connectionBrokenCllbck, locsysID, locsysFrequency, queue):
        logging.info("Initializing tcpClient")
        self.connectiondata = None
        self.connection = None
        self.retry_delay = 1
        self.connectionBrokenCllbck = connectionBrokenCllbck
        self.locsysID = locsysID
        self.locsysFrequency = locsysFrequency
        self.queue = queue
        self.sendLocation = messageHandlerSender(self.queue, Location)
        logging.info("tcpClient initialized")

    def clientConnectionLost(self, connector, reason):
        logging.warning(f"Connection lost with server {self.connectiondata}")
        try:
            self.connectiondata = None
            self.connection = None
            self.connectionBrokenCllbck()
        except Exception as e:
            logging.error(f"Error in clientConnectionLost: {e}")

    def clientConnectionFailed(self, connector, reason):
        logging.warning(f"Connection failed. Retrying in {self.retry_delay} seconds... Possible server down or incorrect IP:port match")
        time.sleep(self.retry_delay)
        connector.connect()

    def buildProtocol(self, addr):
        logging.info("Building protocol")
        conn = SingleConnection(self.queue)
        conn.factory = self
        return conn

    def send_data_to_server(self, message):
        logging.info("Sending data to server")
        self.connection.send_data(message)


# One class is generated for each new connection
class SingleConnection(protocol.Protocol):
    def __init__(self, queue):
        super(SingleConnection, self).__init__()
        self.queue = queue
        
    def connectionMade(self):
        logging.info("Connection made")
        peer = self.transport.getPeer()
        self.factory.connectiondata = peer.host + ":" + str(peer.port)
        self.factory.connection = self
        self.subscribeToLocaitonData(self.factory.locsysID, self.factory.locsysFrequency)
        logging.info(f"Connection with server established: {self.factory.connectiondata}")

    def dataReceived(self, data):
        # logging.info("Data received")
        dat = data.decode()
        tmp_data = dat.replace("}{","}}{{")
        if tmp_data != dat:
            tmp_dat = tmp_data.split("}{")
            dat = tmp_dat[-1]
        da = json.loads(dat)
        self.queue[Position.Queue.value].put(
            {
                "Owner": Position.Owner.value,
                "msgID": Position.msgID.value,
                "msgType": Position.msgType.value,
                "msgValue": da,
            }
        )
        if da["type"] == "location":
            da["id"] = self.factory.locsysID
            self.factory.sendLocation.send(da)
        else:
            logging.info(f"Got message from traffic communication server: {self.factory.connectiondata}")

    def send_data(self, message):
        # logging.info("Sending data")
        msg = json.dumps(message)
        self.transport.write(msg.encode())
    
    def subscribeToLocaitonData(self, id, frequency):
        logging.info("Subscribing to location data")
        # Sends the id you wish to subscribe to and the frequency you want to receive data. Frequency must be between 0.1 and 5. 
        msg = {
            "reqORinfo": "info",
            "type": "locIDsub",
            "locID": id,
            "freq": frequency,
        }
        self.send_data(msg)
    
    def unSubscribeToLocaitonData(self, id, frequency):
        logging.info("Unsubscribing from location data")
        # Unsubscribes from location data. 
        msg = {
            "reqORinfo": "info",
            "type": "locIDubsub",
        }
        self.send_data(msg)