## 模块设计基础

### 考虑因素
1. 输入相关信息：
   1. 目前的 soft priority。
   2. 旧解。所有车的轨迹。
      1. 要知道旧的优先级和旧的轨迹碰撞，才能够知道这个优先级不可行。逆序就可以。
         1. 为什么最优策略不是逆序而是需要一个学习的次序？
   3. 碰撞图. 
   4. 障碍物，起点终点等
2. 输出相关信息
   1. neighbor size
   2. neighbor sequence
      1. 可以用 i j 来表达这两个信息，j-i表达长度，ij表达逆序的片段长度。

### 接口设计
1. `step`函数
   1. 输入 update_sequence. 表示要重新规划的智能体的序列。
2. `getXXX`获取当前的一些信息，包括优先级等。碰撞图。
3. 检测是否成功。

### reward设计
1. 时间和最优性平衡
   1. 时间不应该是迭代次数，而应该是底层的规划的智能体个数。即总共的neighbor size的个数。

### 网络设计

### 一些奇怪的想法
1. lns只能放在把新规划的智能体放在最后，是因为避免大量数量的智能体进行重新规划。否则像pbs一样，就得不断地查找所有因优先级更改的而需要重新规划的智能体。


### 开发计划
1. python boost出了一个奇怪的运行时错误。(done)
   1. 输出：---------------------------------------------------------------------------
   RuntimeError                              Traceback (most recent call last)
   RuntimeError: FATAL: module compiled as little endian, but detected different endianness at runtime
   ---------------------------------------------------------------------------
   ImportError                               Traceback (most recent call last)
   ImportError: numpy.core._multiarray_umath failed to import
   2. 更改成pybind11试试
      1. pybind11 test case.
         1. install. 比较奇妙...新开一个conda环境用conda安装比较快。`conda install -c conda-forge pybind11`
         2. 要在conda环境中进行cmake编译
         3. 在原生python中可以使用，但是在lnsbind的conda中的ipython中不行。（其他的好像可以...）
         4. 
2. TODO 增加一个 load scene时候的随机选择。(低优先级)