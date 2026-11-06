#pragma once

#include "optitrack_client/utils.hpp"

#include <NatNetCAPI.h>
#include <NatNetClient.h>
#include <NatNetTypes.h>

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

class OptitrackClient : public rclcpp::Node {
 public:
  OptitrackClient();
  ~OptitrackClient() override;

  NatNetClient* getNatNetClient() { return g_pClient; }
  void run();

  static constexpr int kMaxQueueSize = 500;
  std::timed_mutex gNetworkQueueMutex;
  std::deque<MocapFrameWrapper> gNetworkQueue;

 private:
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr _marker_publisher;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _rigid_publisher;
  NatNetClient* g_pClient = nullptr;
  bool gUpdatedDataDescriptions = false;
  bool gNeedUpdatedDataDescriptions = true;
  sDataDescriptions* g_pDataDefs = nullptr;
  std::map<int, int> g_AssetIDtoAssetDescriptionOrder;
  std::map<int, std::string> g_AssetIDtoAssetName;

  void DrainNetworkQueue();
};
