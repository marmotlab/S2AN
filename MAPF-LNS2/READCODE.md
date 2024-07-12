## 代码阅读

### paper阅读
1. 根据githu issue里的回复，lns采用的全是soft obstacles. 只有eecbs采用hard obstacles。即对于所有其他智能体障碍物，都是能避撞就避撞。
2. 邻域的选择方法分为三种，在initLns中实现了。
   1. 随机选择
   2. 根据碰撞选择 `InitLNS::generateNeighborByCollisionGraph()`
   3. failure based. 根据target collision 和 run over进行选择。`InitLNS::generateNeighborByTarget()`
   4. 融合以上三种方法
3. 求解过程(详情看###代码阅读章节)
   1. 选择邻域后，使用pp进行求解
   2. 采用的runpp()。也就是随机生成一个次序，邻域内的智能体按照软约束进行求解。


### 代码阅读
1. 主函数。`driver.cpp`。进入`LNS.CPP`的`lns.run()`，返回success则求解成功。lns.run包括寻找初始解(lns2)，并且不断优化(lns1)的过程。
2. 默认参数：
   1. `init_algo_name`为`PP`，pp里面就是随机选择一个智能体顺序进行求解。
   2. `use_init_lns`为true。如果pp找不到解，就使用init_lns进行求解。lns2的算法主要在`initLNS.cpp`中实现
3. `initLNS.cpp`
   1. 关键变量：`path_table`. `vector<vector<list<int>>>` key为location, time，value是此时此刻在这个位置的智能体的id.
   2. 在构造函数时，进行空的初始化
   3. `getInitialSolution`相当于pp。已有轨迹的直接塞入path_table，没有轨迹的当做neighbor，随机次序然后进行sipps。最后把生成的轨迹放进path_table里。
   4. 每次在pp规划前，总会删除neighbor的所有智能体的轨迹。然后shuffle邻域里面智能体的次序，再逐个智能体进行sipss,并且添加已规划智能体的轨迹入path_table中。