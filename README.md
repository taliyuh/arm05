# arm05_search

Na hoście:

```bash
docker cp arm05_search ARM_05:/arm_ws/src/
docker cp arm05_search_interfaces ARM_05:/arm_ws/src/
```

W kontenerze (`docker exec -it ARM_05 bash`):

```bash
apt-get update && apt-get install -y ros-jazzy-py-trees-ros
cd /arm_ws && colcon build --symlink-install --packages-select arm05_search_interfaces arm05_search
source install/setup.bash
```

Uruchomienie (każdy terminal w kontenerze, po `source /arm_ws/install/setup.bash`):

```bash
ros2 launch arm05_sim world.launch.py aruco_seed:=42
ros2 launch arm05_search search.launch.py
ros2 run arm05_search evaluate_client.py 5 900
```
