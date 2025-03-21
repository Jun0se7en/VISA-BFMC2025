import json
from twisted.internet import protocol

from src.utils.messages.allMessages import (
    Brake,
    Control,
    EngineRun,
    Record,
    SpeedMotor,
    SteerMotor,
)

# One class is generated for each new connection
class SingleConnection(protocol.Protocol):
    # ================================= CONNECTION MADE =====+================================
    def connectionMade(self):
        peer = self.transport.getPeer()
        self.factory.connectiondata = peer.host + ":" + str(peer.port)
        self.factory.connection = self
        self.connected = False
        print("Attempting connection by :", self.factory.connectiondata)

    # ================================= CONNECTION LOST ======================================
    def connectionLost(self, reason):
        print("Connection lost with ", self.factory.connectiondata, " due to: ", reason)
        self.factory.isConnected = False
        self.factory.isConnected = False
        self.factory.connectiondata = None
        self.factory.connection = None

    # ================================== DATA RECEIVED =======================================
    def dataReceived(self, data):
        if self.factory.isConnected == False:
            pswd = data.decode()
            if pswd == "Ala-Bala":
                self.factory.isConnected = True
                print(
                    "Connected with ", self.factory.connectiondata, " : ", data.decode()
                )
            else:
                print(
                    "Connection attempted failed with incorrect password ",
                    self.factory.connectiondata,
                )
                self.factory.isConnected = False
                self.factory.connectiondata = None
                self.transport.loseConnection()
                self.factory.connection = None
        else:
            try:
                dataJSON = json.loads(data.decode())
            except:
                dataJSON = {"action": "except"}
            if dataJSON["action"] == "startEngine":
                self.factory.queues[EngineRun.Queue.value].put(
                    {
                        "Owner": EngineRun.Owner.value,
                        "msgID": EngineRun.msgID.value,
                        "msgType": EngineRun.msgType.value,
                        "msgValue": dataJSON["value"],
                    }
                )
                print(
                    {
                        "Owner": EngineRun.Owner.value,
                        "msgID": EngineRun.msgID.value,
                        "msgType": EngineRun.msgType.value,
                        "msgValue": True,
                    }
                )
            elif dataJSON["action"] == "brake":
                self.factory.queues[Brake.Queue.value].put(
                    {
                        "Owner": Brake.Owner.value,
                        "msgID": Brake.msgID.value,
                        "msgType": Brake.msgType.value,
                        "msgValue": dataJSON["value"],
                    }
                )
            elif dataJSON["action"] == "speed":
                self.factory.queues[SpeedMotor.Queue.value].put(
                    {
                        "Owner": SpeedMotor.Owner.value,
                        "msgID": SpeedMotor.msgID.value,
                        "msgType": SpeedMotor.msgType.value,
                        "msgValue": dataJSON["value"],
                    }
                )
            elif dataJSON["action"] == "steer":
                self.factory.queues[SteerMotor.Queue.value].put(
                    {
                        "Owner": SteerMotor.Owner.value,
                        "msgID": SteerMotor.msgID.value,
                        "msgType": SteerMotor.msgType.value,
                        "msgValue": dataJSON["value"],
                    }
                )
            elif dataJSON["action"] == "startRecord":
                self.factory.queues[Record.Queue.value].put(
                    {
                        "Owner": Record.Owner.value,
                        "msgID": Record.msgID.value,
                        "msgType": Record.msgType.value,
                        "msgValue": dataJSON["value"],
                    }
                )
            elif dataJSON["action"] == "STS":
                self.factory.queues[Record.Queue.value].put(
                    {
                        "Owner": Control.Owner.value,
                        "msgID": Control.msgID.value,
                        "msgType": Control.msgType.value,
                        "msgValue": dataJSON["value"],
                    }
                )

    # ===================================== SEND DATA ==========================================
    def send_data(self, messageValue, messageType, messageOwner, messageId):
        """This function will send firstly an encoded message as an int represented in one byte after that it will send the lenght of the message and the message."""
        self.transport.write(
            self.factory.encoder[(messageType, messageOwner, messageId)].to_bytes(
                1, byteorder="big"
            )
        )
        self.transport.write(
            len(messageValue.encode("utf-8")).to_bytes(4, byteorder="big")
        )  # send size of image
        self.transport.write(messageValue.encode("utf-8"))  # send image data


from src.utils.messages.allMessages import (
    Cars,
    EnableButton,
    Location,
    Recording,
    Semaphores,
    SignalRunning,
    serialCamera,
)


# The server itself. Creates a new Protocol for each new connection and has the info for all of them.
class FactoryDealer(protocol.Factory):
    # ======================================= INIT =============================================
    def __init__(self, queues):
        self.connection = None
        self.isConnected = False
        self.connectiondata = None
        self.queues = queues
        self.encoder = {
            (
                serialCamera.msgType.value,
                serialCamera.Owner.value,
                serialCamera.msgID.value,
            ): 1,
            "add another value in table(use the same structure )": 2,
            (
                Cars.msgType.value,
                Cars.Owner.value,
                Cars.msgID.value,
            ): 3,
            (
                Semaphores.msgType.value,
                Semaphores.Owner.value,
                Semaphores.msgID.value,
            ): 4,
            (
                EnableButton.msgType.value,
                EnableButton.Owner.value,
                EnableButton.msgID.value,
            ): 5,
            (
                SignalRunning.msgType.value,
                SignalRunning.Owner.value,
                SignalRunning.msgID.value,
            ): 6,
            (
                Recording.msgType.value,
                Recording.Owner.value,
                Recording.msgID.value,
            ): 7,
            (
                Location.msgType.value,
                Location.Owner.value,
                Location.msgID.value,
            ): 8,
        }

    def send_data_to_client(self, messageValue, messageType, messageOwner, messageId):
        """This function will try to send the information only if there is a connection between Demo and raspberry PI."""
        if self.isConnected == True:
            self.connection.send_data(
                messageValue, messageType, messageOwner, messageId
            )
        else:
            print("Client not connected")

    # ================================== BUILD PROTOCOL ========================================
    def buildProtocol(self, addr):
        conn = SingleConnection()
        conn.factory = self
        return conn

    # =================================== AUXILIARY ============================================
    def doStart(self):
        print("Start factory")

    def doStop(self):
        print("Stop factory")
