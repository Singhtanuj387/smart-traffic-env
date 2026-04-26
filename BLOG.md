# Smart Traffic Manager Environment

Ever stuck in a *traffic jam* wasting hours of your precious time and energy- quite common across the world. Or the worse part- seeing an emergency vehicle stuck in traffic melts the heart of everyone.
So inspired by these issues our OpenEnv environment comes to rescue.

## The Issues solved
Our environment builds the situations on top of the following problems:

 - peak-hour congestion
 - emergency vehicle priority
 - road block and detour handling
 - weather-related slowdown
 - event-based traffic surges
 - directional imbalance
 - cascading congestion failure
 - long-horizon planning
 - and recovery from early mistakes...........
which enables the agent to learn to decide the traffic lights of the city.

## How it Works
Our environment models a typical city as a 9 x 9 grid and each square (i.e. 81 squares) represents a crossroad of the city. The environment treats each crossroad of the city (each square of the grid) as a *separate agent* which controls the traffic lights of that crossroad.
![A typical Indian Crossroad](https://www.hindustantimes.com/ht-img/img/2024/09/16/1600x900/Proponents-of-signals-argue-that-they-not-just-hel_1726482453680.jpg)
In this approach:
firstly the agent A1 clears a path to a particular direction by controlling the traffic lights and then A2 decides according to A1s action and decides which direction's traffic should be given the way and so on A3 decides and this continues........

#### why cannot every agent decide simultaneously which path to clear and not. 
Initially we followed that conventional intuition of our minds only but on further thinking and prompting we explored that the key issue here is of non-stationarity (i.e. agents keep changing each other’s environment) and hence traffic control is a multi agent problem. So we shifted towards the sequential approach as mentioned above.
| | Simultaneous switching of traffic lights|Simultaneous switching of traffic lights  | |
|--|--|-- |-- |
|  Waiting times at traffic lights|High |Low |
|  Congestion Spikes|High |Low |

## Rewards

 -   **Correctness → Total Throughput**  
    Measures whether vehicles are flowing efficiently through the system (i.e., cars are successfully passing through intersections).
-   **Coverage → Network Efficiency**  
    Ensures balanced utilization across the entire 9×9 grid (i.e., all lanes are being served fairly, avoiding neglect of any region).
-   **Repetition → Phase Stability (Flicker Control)**  
    Prevents excessive or rapid traffic light switching (i.e., avoids unstable behavior and flickering signals).

*Made with ❤️ by
Tanuj and Madhav*