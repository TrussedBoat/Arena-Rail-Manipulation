"""Isaac Sim ROS 2 simulation-clock setup shared by Arena entrypoints."""


def configure_ros2_sim_time(stage) -> int:
    """Publish `/clock` and make camera helper timestamps use simulation time."""
    import omni.graph.core as og

    graph_path = "/ArenaRos2Clock"
    keys = og.Controller.Keys
    graph_spec = {"graph_path": graph_path, "evaluator_name": "execution"}
    og.Controller.edit(
        graph_spec,
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimulationTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimulationTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", "clock"),
            ],
        },
    )

    changed = 0
    for prim in stage.Traverse():
        attribute = prim.GetAttribute("inputs:useSystemTime")
        if attribute.IsValid():
            attribute.Set(False)
            changed += 1
    return changed
