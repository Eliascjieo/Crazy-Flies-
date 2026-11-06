#include "optitrack_client/optitrack_client.hpp"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

void NATNET_CALLCONV DataHandler(sFrameOfMocapData* data, void* pUserData) {
  OptitrackClient* oClient = static_cast<OptitrackClient*>(pUserData);
  if (!oClient) return;
  NatNetClient* pClient = oClient->getNatNetClient();
  if (!pClient) return;

  // The 'data' ptr is managed by NatNet and cannot be used outside this
  // callback. Copy it before pushing onto our queue.
  std::shared_ptr<sFrameOfMocapData> pDataCopy =
      std::make_shared<sFrameOfMocapData>();
  NatNet_CopyFrame(data, pDataCopy.get());

  MocapFrameWrapper f;
  f.data = pDataCopy;
  f.clientLatencyMillisec =
      pClient->SecondsSinceHostTimestamp(data->CameraMidExposureTimestamp) *
      1000.0;
  f.transitLatencyMillisec =
      pClient->SecondsSinceHostTimestamp(data->TransmitTimestamp) * 1000.0;

  if (oClient->gNetworkQueueMutex.try_lock_for(std::chrono::milliseconds(5))) {
    oClient->gNetworkQueue.push_back(f);

    while ((int)oClient->gNetworkQueue.size() > oClient->kMaxQueueSize) {
      f = oClient->gNetworkQueue.front();
      NatNet_FreeFrame(f.data.get());
      oClient->gNetworkQueue.pop_front();
    }
    oClient->gNetworkQueueMutex.unlock();
  } else {
    NatNet_FreeFrame(pDataCopy.get());
    printf("\nFrame dropped (Frame : %d)\n", f.data->iFrame);
  }
}

OptitrackClient::OptitrackClient() : rclcpp::Node("optitrack_client") {
  _marker_publisher = create_publisher<geometry_msgs::msg::PointStamped>(
      "optitrack/marker", rclcpp::SensorDataQoS());
  _rigid_publisher = create_publisher<geometry_msgs::msg::PoseStamped>(
      "optitrack/rigid_body", rclcpp::SensorDataQoS());

  unsigned char ver[4];
  NatNet_GetVersion(ver);
  RCLCPP_INFO(get_logger(), "NatNet Custom Client (NatNet ver. %d.%d.%d.%d)",
              ver[0], ver[1], ver[2], ver[3]);

  NatNet_SetLogCallback(MessageHandler);
  g_pClient = new NatNetClient();
  g_pClient->SetFrameReceivedCallback(DataHandler, this);
}

OptitrackClient::~OptitrackClient() {
  if (g_pClient) {
    g_pClient->Disconnect();
    delete g_pClient;
    g_pClient = nullptr;
  }
  if (g_pDataDefs) {
    NatNet_FreeDescriptions(g_pDataDefs);
    g_pDataDefs = nullptr;
  }
}

void OptitrackClient::run() {
  const int kMaxDescriptions = 10;
  sNatNetDiscoveredServer servers[kMaxDescriptions];
  int actualNumDescriptions = kMaxDescriptions;
  NatNet_BroadcastServerDiscovery(servers, &actualNumDescriptions);

  if (actualNumDescriptions == 0) {
    throw std::runtime_error("[Error] No NatNet servers found");
  }
  const sNatNetDiscoveredServer& discoveredServer = servers[0];

  char g_discoveredMulticastGroupAddr[kNatNetIpv4AddrStrLenMax] =
      NATNET_DEFAULT_MULTICAST_ADDRESS;
  sNatNetClientConnectParams g_connectParams;
  snprintf(g_discoveredMulticastGroupAddr,
           sizeof g_discoveredMulticastGroupAddr,
           "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8 "",
           discoveredServer.serverDescription.ConnectionMulticastAddress[0],
           discoveredServer.serverDescription.ConnectionMulticastAddress[1],
           discoveredServer.serverDescription.ConnectionMulticastAddress[2],
           discoveredServer.serverDescription.ConnectionMulticastAddress[3]);

  g_connectParams.connectionType =
      discoveredServer.serverDescription.ConnectionMulticast
          ? ConnectionType_Multicast
          : ConnectionType_Unicast;
  g_connectParams.serverCommandPort = discoveredServer.serverCommandPort;
  g_connectParams.serverDataPort =
      discoveredServer.serverDescription.ConnectionDataPort;
  g_connectParams.serverAddress = discoveredServer.serverAddress;
  g_connectParams.localAddress = discoveredServer.localAddress;
  g_connectParams.multicastAddress = g_discoveredMulticastGroupAddr;

  RCLCPP_INFO(get_logger(), "[Info] Connecting to %s",
              g_connectParams.serverAddress);

  int iResult = ConnectClient(g_pClient, g_connectParams);
  if (iResult != ErrorCode_OK) {
    throw std::runtime_error(
        "Error initializing client. See log for details. Exiting.");
  }
  RCLCPP_INFO(get_logger(), "Client initialized and ready.");

  gUpdatedDataDescriptions = UpdateDataDescriptions(
      g_pClient, g_pDataDefs, g_AssetIDtoAssetDescriptionOrder,
      g_AssetIDtoAssetName);
  if (!gUpdatedDataDescriptions) {
    RCLCPP_ERROR(get_logger(),
                 "Unable to retrieve Data Descriptions from Motive.");
  }

  RCLCPP_INFO(get_logger(),
              "Client is connected to server and listening for data...");

  while (rclcpp::ok()) {
    if (gNeedUpdatedDataDescriptions) {
      gUpdatedDataDescriptions = UpdateDataDescriptions(
          g_pClient, g_pDataDefs, g_AssetIDtoAssetDescriptionOrder,
          g_AssetIDtoAssetName);
      if (gUpdatedDataDescriptions) {
        gNeedUpdatedDataDescriptions = false;
      }
    }

    DrainNetworkQueue();
    rclcpp::spin_some(this->get_node_base_interface());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void OptitrackClient::DrainNetworkQueue() {
  std::deque<MocapFrameWrapper> displayQueue;
  if (gNetworkQueueMutex.try_lock_for(std::chrono::milliseconds(5))) {
    for (MocapFrameWrapper f : gNetworkQueue) {
      displayQueue.push_back(f);
    }
    gNetworkQueue.clear();
    gNetworkQueueMutex.unlock();
  }

  for (MocapFrameWrapper f : displayQueue) {
    sFrameOfMocapData* data = f.data.get();

    if (gNeedUpdatedDataDescriptions) {
      RCLCPP_INFO(get_logger(), "Waiting for updated asset list");
      break;
    }

    bool bTrackedModelsChanged = ((data->params & 0x02) != 0);
    if (bTrackedModelsChanged) {
      RCLCPP_INFO(get_logger(),
                  "Motive asset list changed.  Requesting new data descriptions.");
      gNeedUpdatedDataDescriptions = true;
      break;
    }

    const rclcpp::Time stamp = this->now();

    for (int i = 0; i < data->nRigidBodies; i++) {
      bool bTrackingValid = data->RigidBodies[i].params & 0x01;
      if (!bTrackingValid) continue;

      int streamingID = data->RigidBodies[i].ID;

      geometry_msgs::msg::PoseStamped msg;
      msg.header.stamp = stamp;
      auto name_it = g_AssetIDtoAssetName.find(streamingID);
      msg.header.frame_id = (name_it != g_AssetIDtoAssetName.end())
                                ? name_it->second
                                : std::to_string(streamingID);
      msg.pose.position.x = data->RigidBodies[i].x;
      msg.pose.position.y = data->RigidBodies[i].y;
      msg.pose.position.z = data->RigidBodies[i].z;
      msg.pose.orientation.x = data->RigidBodies[i].qx;
      msg.pose.orientation.y = data->RigidBodies[i].qy;
      msg.pose.orientation.z = data->RigidBodies[i].qz;
      msg.pose.orientation.w = data->RigidBodies[i].qw;
      _rigid_publisher->publish(msg);
    }

    for (int i = 0; i < data->nLabeledMarkers; i++) {
      sMarker marker = data->LabeledMarkers[i];

      int modelID, markerID;
      NatNet_DecodeID(marker.ID, &modelID, &markerID);

      geometry_msgs::msg::PointStamped marker_msg;
      marker_msg.header.stamp = stamp;
      marker_msg.header.frame_id = std::to_string(markerID);
      marker_msg.point.x = marker.x;
      marker_msg.point.y = marker.y;
      marker_msg.point.z = marker.z;
      _marker_publisher->publish(marker_msg);
    }
  }

  for (MocapFrameWrapper f : displayQueue) {
    NatNet_FreeFrame(f.data.get());
  }
  displayQueue.clear();
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<OptitrackClient>();
    node->run();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("optitrack_client"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
