#include "SippsSolver.h"
#include "common.h"
#include <queue>
#include <algorithm>
#include "GCBS.h"
#include "PBS.h"

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

SolverWrapper::SolverWrapper(
    py::array_t<double> world, int world_len, int world_wid, 
    py::array_t<double> goals, int num_agents, int max_num_iter, 
    int neighbor_size, int screen, double time_limit): 
    m_world_len(world_len), m_world_wid(world_wid),
    num_agents(num_agents)
    {
        py::buffer_info info = world.request();
        auto ptr = static_cast<double *>(info.ptr);

        for ( int i = 0 ; i < world_len; i++){
            for ( int j = 0 ; j < world_wid; j++){
                m_world.emplace_back(ptr[i*world_wid +j]);
            }
        }

        obstacle_map.reserve(world_len* world_wid);
        // construct the map. 1 obstacle. 0 free.
        for ( int i = 0; i < world_len * world_wid; i++){
            // my_map : 1: obstacle, 0: free.
            // ptr: -1: obstacle. 0: free. >0 agent start postion.
            obstacle_map.emplace_back(ptr[i]<0);
        }

        start_locations.resize(num_agents);
        for ( int i = 0; i < world_len * world_wid; i++){
            if (ptr[i] > 0){
                int agent_id = ptr[i] - 1;
                start_locations[agent_id] = i;
            }
        }

        // print the start locations
        if (screen > 1){
            std::cout << "start location: \n";
            for (int i = 0; i < num_agents; i++){
                std::cout << start_locations[i] << ", ";
            }
            std::cout << std::endl;
        }


        goal_locations.resize(num_agents);
        info = goals.request();
        auto array = static_cast<double *>(info.ptr);
        for ( int i = 0 ; i < world_len; i++){
            for ( int j = 0 ; j < world_wid; j++){
                m_goals.emplace_back(array[i* world_wid + j]);
                if (array[i* world_wid + j] > 0){
                    int agent_id = array[i* world_wid + j] - 1;
                    goal_locations[agent_id] = i* world_wid + j;
                }
            }
        }

        // print the goal locations
        if (screen > 1){
            std::cout << "goal location: \n";
            for (int i = 0; i < num_agents; i++){
                std::cout << goal_locations[i] << ", ";
            }
            std::cout << std::endl;
        }

        instance = Instance(num_agents, obstacle_map, start_locations,
            goal_locations, world_len, world_wid);

        agents.reserve(num_agents);
        bool use_sipp = true;
        for (int i = 0; i < num_agents; i++)
            agents.emplace_back(instance, i, use_sipp);
        string replan_algo_name("PP");
        string init_destory_name("Target");   // this should not be used anymore
        // int neighbor_size = 8;          // this should not be used anymore
        sipps_solver = new SippsSolver(instance, agents, time_limit,
                replan_algo_name,init_destory_name, neighbor_size, 
                max_num_iter, screen);
        

}

void SolverWrapper::calcInitialSolution(py::array_t<double> init_seq, int num_agents){
    
    py::buffer_info info = init_seq.request();
    auto array = reinterpret_cast<double*>(info.ptr);
    vector<int> agents_seq;    
    for ( int i = 0 ; i < num_agents; i++){
        agents_seq.emplace_back(int(array[i]));
    }


    sipps_solver->getInitialSolution(agents_seq);
}

void SolverWrapper::step(py::array_t<double> replan_seq, int num_seq){
    py::buffer_info info = replan_seq.request();

    auto array = reinterpret_cast<double*>(info.ptr);

    vector<int> agents_seq;
    
    for ( int i = 0 ; i < num_seq; i++){
        agents_seq.emplace_back(int(array[i]));
    }

    sipps_solver->step(agents_seq);
    
}


SippsSolver::SippsSolver(const Instance& instance, vector<Agent>& agents, double time_limit,
         const string & replan_algo_name, const string & init_destory_name, int neighbor_size,
         int max_num_iter, int screen) :
         BasicLNS(instance, time_limit, neighbor_size, screen), agents(agents), replan_algo_name(replan_algo_name),
         path_table(instance.map_size, agents.size()), collision_graph(agents.size()), goal_table(instance.map_size, -1),
         num_of_agents(agents.size()), path_table_agents(num_of_agents, false),
         max_num_iter(max_num_iter)
{
    replan_time_limit = time_limit;
    if (init_destory_name == "Adaptive")
    {
        ALNS = true;
        destroy_weights.assign(INIT_COUNT, 1);
        decay_factor = 0.05;
        reaction_factor = 0.05;
    }
    else if (init_destory_name == "Target")
        init_destroy_strategy = TARGET_BASED;
    else if (init_destory_name == "Collision")
        init_destroy_strategy = COLLISION_BASED;
    else if (init_destory_name == "Random")
        init_destroy_strategy = RANDOM_BASED;
    else
    {
        cerr << "Init Destroy heuristic " << init_destory_name << " does not exists. " << endl;
        exit(-1);
    }

    for (auto& i:agents) {
        goal_table[i.path_planner->goal_location] = i.id;
    }

    for (int a = 0; a < num_of_agents; a++){
        set<int> A_target;
        agents[a].path_planner->findMinimumSetofColldingTargets(goal_table,A_target);// generate non-wait path and collect A_target
        if ( !A_target.empty() ){
            for (auto& t : A_target){
                target_edge_index.emplace_back(a, t);
            }
        }
    }


}

Eigen::MatrixXi SippsSolver::getTragetEdgeIndex() const{
    Eigen::MatrixXi edge_index(target_edge_index.size(), 2);
    
    int count = 0;
    for (auto ei : target_edge_index){
        edge_index(count, 0) = ei.first;
        edge_index(count++, 1) = ei.second;
    }

    return edge_index;
}

void SippsSolver::updatePriorities(const vector<int>& replan_seq){
    if (priorities.empty()){
        priorities.assign(replan_seq.begin(), replan_seq.end());
    }
    else{
        set<int> end_priorities(replan_seq.begin(), replan_seq.end());
        vector<int> last_priorities(priorities.begin(), priorities.end());
        priorities.clear();
        for (int i = 0 ; i < last_priorities.size(); i++){
            if ( end_priorities.find(last_priorities[i]) != end_priorities.end()){
                continue;
            }
            priorities.emplace_back(last_priorities[i]);
        }
        for ( auto id : replan_seq){
            priorities.emplace_back(id);
        }
    }
}

bool SippsSolver::step(const vector<int>& replan_seq){
    updatePriorities(replan_seq);
    
    neighbor.agents.assign(replan_seq.begin(), replan_seq.end());
    if (screen >= 2)
        cout << "Input " << neighbor.agents.size() << " neighbors" << endl;

    vector<Path*> paths(agents.size());
    for (auto i = 0; i < agents.size(); i++)
        paths[i] = &agents[i].path;

    assert(instance.validateSolution(paths, sum_of_costs, num_of_colliding_pairs));

    // solve the results.

    // get colliding pairs
    neighbor.old_colliding_pairs.clear();
    for (int a : neighbor.agents)
    {
        for (auto j: collision_graph[a])
        {
            neighbor.old_colliding_pairs.emplace(min(a, j), max(a, j));
        }
    }
    // remove this by checking the results
    if (neighbor.old_colliding_pairs.empty()) // no need to replan
    {
        return false;
    }

    // store the neighbor information
    neighbor.old_paths.resize(neighbor.agents.size());
    neighbor.old_sum_of_costs = 0;
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
    {
        int a = neighbor.agents[i];
        if (replan_algo_name == "PP" || neighbor.agents.size() == 1)
            neighbor.old_paths[i] = agents[a].path;
        path_table.deletePath(neighbor.agents[i]);
        neighbor.old_sum_of_costs += (int) agents[a].path.size() - 1;
    }
    if (screen >= 2)
    {
        cout << "Neighbors: ";
        for (auto a : neighbor.agents)
            cout << a << ", ";
        cout << endl;
        cout << "Old colliding pairs (" << neighbor.old_colliding_pairs.size() << "): ";
        for (const auto & p : neighbor.old_colliding_pairs)
        {
            cout << "(" << p.first << "," << p.second << "), ";
        }
        cout << endl;

    }
    bool succ=false;
    if (replan_algo_name == "PP" || neighbor.agents.size() == 1)
        succ = runPP();
    else if (replan_algo_name == "GCBS")
        succ = runGCBS();
    else if (replan_algo_name == "PBS")
        succ = runPBS();
    else
    {
        cerr << "Wrong replanning strategy" << endl;
        exit(-1);
    }

    if (screen >= 2)
        cout << "New colliding pairs = " << neighbor.colliding_pairs.size() << endl;
    if (succ) // update collision graph
    {
        num_of_colliding_pairs += (int)neighbor.colliding_pairs.size() - (int)neighbor.old_colliding_pairs.size();
        for(const auto& agent_pair : neighbor.old_colliding_pairs)
        {
            collision_graph[agent_pair.first].erase(agent_pair.second);
            collision_graph[agent_pair.second].erase(agent_pair.first);
        }
        for(const auto& agent_pair : neighbor.colliding_pairs)
        {
            collision_graph[agent_pair.first].emplace(agent_pair.second);
            collision_graph[agent_pair.second].emplace(agent_pair.first);
        }
        if (screen >= 2)
            printCollisionGraph();
    }
    runtime = ((fsec)(Time::now() - start_time)).count();
    sum_of_costs += neighbor.sum_of_costs - neighbor.old_sum_of_costs;
    if (screen >= 1)
        cout << "Iteration " << iteration_stats.size() << ", "
                << "group size = " << neighbor.agents.size() << ", "
                << "colliding pairs = " << num_of_colliding_pairs << ", "
                << "solution cost = " << sum_of_costs << ", "
                << "remaining time = " << time_limit - runtime << endl;
    iteration_stats.emplace_back(neighbor.agents.size(), sum_of_costs, runtime, replan_algo_name,
                                    0, num_of_colliding_pairs);

    printResult();
    return (num_of_colliding_pairs == 0);

}

Eigen::VectorXi SippsSolver::getPriorities() const{
    Eigen::VectorXi priorities_seq(num_of_agents);
    int i =0;
    for (auto& p : priorities){
        priorities_seq[i++] = p;
    }
    for ( ;i<num_of_agents; i++){
        priorities_seq[i] = -1;
    }
    return priorities_seq;
}

Eigen::MatrixXi SippsSolver::getCollisionEdgeIndex() const{
    int num_edges = 0;
    for (auto out_degree : collision_graph){
        num_edges += out_degree.size();
    }
    if (screen>2)
        cout<< "num edges: "<< num_edges<<std::endl;

    Eigen::MatrixXi edge_index(num_edges, 2);

    int count = 0;
    for (int i = 0 ; i < collision_graph.size(); i++){
        for ( auto aj : collision_graph[i]){
            edge_index(count, 0) = i;
            edge_index(count++, 1) = aj;
        }
    }
    return edge_index;
}

Eigen::MatrixXi SippsSolver::getSolutions() const{
    int max_t = 0; 
    int num_of_agents = agents.size();
    for ( int i = 0; i < num_of_agents; i++){
        if (agents[i].path.size() > max_t)
            max_t = agents[i].path.size() ;
    }

    Eigen::MatrixXi res(num_of_agents, max_t * 2);

    if(screen > 2){
        std::cout << "solution shape: [" << num_of_agents<< ", " 
            << max_t << ", 2]. " <<std::endl;
    }

    for ( int i = 0 ; i < num_of_agents; i++){
        int t = 0;  // time index.
        for (const auto & p : agents[i].path ){
            res(i, 2*t) = 
                instance.getRowCoordinate(p.location);
            res(i, 2*t+1) = 
                instance.getColCoordinate(p.location);
            t++;
        }
        for (; t < max_t; t++){
            res(i, 2*t) = -1;
            res(i, 2*t + 1) = -1;
        }
    }

    // if (screen > 2){
    //     std::cout << "solutions:\n";
    //     for ( int i= 0 ; i < num_of_agents; i++){
    //         for ( int t = 0 ; t < max_t; t++){
    //             if (res(i, 2*t) == -1) break;
    //             cout<< "(" << res(i, 2*t) << ","
    //                 << res(i, 2*t +1)<< ")->";
    //         }
    //         cout << std::endl;        
    //     }
    // }


    return res;    
}

bool SippsSolver::validateSolutions(py::array_t<int> paths_np, int solution_soc) const{


    py::buffer_info info = paths_np.request();
    auto array = static_cast<int *>(info.ptr);


    int num_agents = info.shape[0];
    int max_t = info.shape[1];

    vector<vector<int>> paths;
    // paths.reserve(num_of_agents);

    for ( int i = 0 ; i < num_agents; i++){
        vector<int> path;
        // path.reserve(max_t);

        for ( int j = 0 ; j < max_t; j++){
            int pos = int(array[i*max_t +j]);
            if (pos == -1){
                break;
            }
            path.emplace_back(pos);
        }
        paths.emplace_back(path);
    }

    // check whether the paths are feasible
	size_t soc = 0;
    
	for (int a1 = 0; a1 < num_of_agents; a1++)
	{
		soc += paths[a1].size() - 1;
		for (int a2 = a1 + 1; a2 < num_of_agents; a2++)
		{
			size_t min_path_length = paths[a1].size() < paths[a2].size() ? paths[a1].size() : paths[a2].size();
			for (size_t timestep = 0; timestep < min_path_length; timestep++)
			{
				int loc1 = paths[a1].at(timestep);
				int loc2 = paths[a2].at(timestep);
				if (loc1 == loc2)
				{
					cout << "Agents " << a1 << " and " << a2 << " collides at " << loc1 << " at timestep " << timestep << endl;
					return false;
				}
				else if (timestep < min_path_length - 1
					&& loc1 == paths[a2].at(timestep + 1)
					&& loc2 == paths[a1].at(timestep + 1))
				{
					cout << "Agents " << a1 << " and " << a2 << " collides at (" <<
						loc1 << "-." << loc2 << ") at timestep " << timestep << endl;
					return false;
				}
			}
			if (paths[a1].size() != paths[a2].size())
			{
				int a1_ = paths[a1].size() < paths[a2].size() ? a1 : a2;
				int a2_ = paths[a1].size() < paths[a2].size() ? a2 : a1;
				int loc1 = paths[a1_].back();
				for (size_t timestep = min_path_length; timestep < paths[a2_].size(); timestep++)
				{
					int loc2 = paths[a2_].at(timestep);
					if (loc1 == loc2)
					{
						cout << "Agents " << a1 << " and " << a2 << " collides at " << loc1 << " at timestep " << timestep << endl;
						return false; // It's at least a semi conflict			
					}
				}
			}
		}
	}
    // solution soc == -1. dont check the soc.
	if (solution_soc!=-1 && (int)soc != solution_soc)
	{
		cout << "The solution cost is wrong!" << endl;
		return false;
	}
	return true;
}

bool SippsSolver::solveSingleAgent(int id, vector<int>& higher_agnets) {
    // have collision 是和这个智能体有关的新碰撞产生了。之前也许有旧的碰撞
    // sequentially solve the problems.
    bool have_collision = false;

    assert(std::find(priorities.begin(), priorities.end(), id) == priorities.end());
    priorities.emplace_back(id);

    // check and update the path table
    vector<bool> higer_table(num_of_agents, false);
    for (auto ha : higher_agnets){
        higer_table[ha] = true;
    }
    assert(higer_table == path_table_agents);
    // for ( int i = 0; i < num_of_agents; i++){
    //     if (higer_table[i] == path_table_agents[i]){
    //         continue;
    //     }
    //     if (higer_table[i]){
    //         path_table.insertPath(i, agents[i].path);
    //     }
    //     else{
    //         path_table.deletePath(i);
    //     }
    //     path_table_agents[i] = higer_table[i];
    // }
    
    // use sipps to solve the results
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, nullptr, &path_table);
    neighbor.sum_of_costs = 0;
    neighbor.colliding_pairs.clear();

    agents[id].path = agents[id].path_planner->findPath(constraint_table);
    assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
    if (agents[id].path_planner->num_collisions > 0){
        updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path);
        have_collision = true;        

        // update collision info
        for(const auto& agent_pair : neighbor.colliding_pairs)
        {
            collision_graph[agent_pair.first].emplace(agent_pair.second);
            collision_graph[agent_pair.second].emplace(agent_pair.first);
        }
        num_of_colliding_pairs += agents[id].path_planner->num_collisions;
        
    }
    assert(agents[id].path_planner->num_collisions > 0 or
        !updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path));

    path_table.insertPath(id, agents[id].path);
    path_table_agents[id] = true;
    
    
    return have_collision;
}

int SippsSolver::repairSolutions(){
    // repair the solution within max_num_iter. 
    // if repair is successful, return num_iter, else return -1;
    cout << "enter repair solutions." << endl;
    int num_iter = 0;
    bool succ;
    while ( num_iter < max_num_iter){
        cout << "num iter: " << num_iter << " of total "<< max_num_iter << endl;
        if (screen >= 2)
                printCollisionGraph();
        int neighbor_size_tmp = neighbor_size;
        neighbor_size = min(neighbor_size, int(priorities.size()));
        cout << "neighbor size: " << neighbor_size << endl;

        succ = generateNeighborByCollisionGraph();
        // succ = generateNeighborByTarget();
        cout << "generate succ? " << succ << endl;
        set<int> neighbors_set(neighbor.agents.begin(), neighbor.agents.end());
        // assert(
        //     std::find(neighbors_set.begin(), neighbors_set.end(), priorities.back())
        //     != neighbors_set.end()
        // );

        // neighbors_set.erase(priorities.back());
        vector<int> shuffled_agents(neighbors_set.begin(), neighbors_set.end());

        std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
        neighbor.agents = shuffled_agents;
        // neighbor.agents.insert(neighbor.agents.begin(), priorities.back());


        cout << "run pp.\n";

        // get colliding pairs
        neighbor.old_colliding_pairs.clear();
        for (int a : neighbor.agents)
        {
            for (auto j: collision_graph[a])
            {
                neighbor.old_colliding_pairs.emplace(min(a, j), max(a, j));
            }
        }
        // no need to replan
        assert (!neighbor.old_colliding_pairs.empty()); 

        // store the neighbor information
        neighbor.old_paths.resize(neighbor.agents.size());
        neighbor.old_sum_of_costs = 0;
        for (int i = 0; i < (int)neighbor.agents.size(); i++)
        {
            int a = neighbor.agents[i];
            if (replan_algo_name == "PP" || neighbor.agents.size() == 1)
                neighbor.old_paths[i] = agents[a].path;
            path_table.deletePath(neighbor.agents[i]);
            neighbor.old_sum_of_costs += (int) agents[a].path.size() - 1;
        }
        succ = runPP();
        // succ. update the result.
        cout << "run pp success? " << succ << endl;
        if (succ){
            num_of_colliding_pairs += (int)neighbor.colliding_pairs.size() - (int)neighbor.old_colliding_pairs.size();
            for(const auto& agent_pair : neighbor.old_colliding_pairs)
            {
                collision_graph[agent_pair.first].erase(agent_pair.second);
                collision_graph[agent_pair.second].erase(agent_pair.first);
            }
            for(const auto& agent_pair : neighbor.colliding_pairs)
            {
                collision_graph[agent_pair.first].emplace(agent_pair.second);
                collision_graph[agent_pair.second].emplace(agent_pair.first);
            }
            // update the priority of it.
            PrintPriorities();
            cout << "update priorities.\n";

            updatePriorities(shuffled_agents);
            PrintPriorities();
            if (screen >= 2)
                printCollisionGraph();
        }
        if (screen >= 1)
            cout  << "group size = " << neighbor.agents.size() << ", "
                    << "colliding pairs = " << num_of_colliding_pairs << ", "
                    << "solution cost = " << sum_of_costs << ", "
                    << "remaining time = " << time_limit - runtime << endl;

        neighbor_size = neighbor_size_tmp;
        if(num_of_colliding_pairs == 0) break;
        num_iter++;
    }
    
    int res = num_iter;
    if (num_of_colliding_pairs != 0)
        res = -1;
    return res;
}

bool SippsSolver::runGCBS()
{
    vector<SingleAgentSolver*> search_engines;
    search_engines.reserve(neighbor.agents.size());
    for (int i : neighbor.agents)
    {
        search_engines.push_back(agents[i].path_planner);
    }

    // build path tables
    vector<PathTable> path_tables(neighbor.agents.size(), PathTable(instance.map_size));
    for (int i = 0; i < (int)neighbor.agents.size(); i++)
    {
        int agent_id = neighbor.agents[i];
        for (int j = 0; j < instance.getDefaultNumberOfAgents(); j++)
        {
            if (j != agent_id and collision_graph[agent_id].count(j) == 0)
                path_tables[i].insertPath(j, agents[j].path);
        }
    }

    GCBS gcbs(search_engines, screen - 1, &path_tables);
    gcbs.setDisjointSplitting(false);
    gcbs.setBypass(true);
    gcbs.setTargetReasoning(true);

    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime;
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    gcbs.solve(T);
    if (gcbs.best_node->colliding_pairs < (int) neighbor.old_colliding_pairs.size()) // accept new paths
    {
        auto id = neighbor.agents.begin();
        neighbor.colliding_pairs.clear();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *gcbs.paths[i];
            updateCollidingPairs(neighbor.colliding_pairs, agents[*id].id, agents[*id].path);
            path_table.insertPath(agents[*id].id, agents[*id].path);
            ++id;
        }
        neighbor.sum_of_costs = gcbs.best_node->sum_of_costs;
        return true;
    }
    else // stick to old paths
    {
        if (!neighbor.old_paths.empty())
        {
            for (int id : neighbor.agents)
            {
                path_table.insertPath(agents[id].id, agents[id].path);
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        num_of_failures++;
        return false;
    }
}
bool SippsSolver::runPBS()
{
    vector<SingleAgentSolver*> search_engines;
    search_engines.reserve(neighbor.agents.size());
    vector<const Path*> initial_paths;
    initial_paths.reserve(neighbor.agents.size());
    for (int i : neighbor.agents)
    {
        search_engines.push_back(agents[i].path_planner);
        initial_paths.push_back(&agents[i].path);
    }

    PBS pbs(search_engines, path_table, screen - 1);
    // pbs.setInitialPath(initial_paths);
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime;
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    bool succ = pbs.solve(T, (int)neighbor.agents.size(), neighbor.old_colliding_pairs.size());
    if (succ and pbs.best_node->getCollidingPairs() < (int) neighbor.old_colliding_pairs.size()) // accept new paths
    {
        auto id = neighbor.agents.begin();
        neighbor.colliding_pairs.clear();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *pbs.paths[i];
            updateCollidingPairs(neighbor.colliding_pairs, agents[*id].id, agents[*id].path);
            path_table.insertPath(agents[*id].id);
            ++id;
        }
        assert(neighbor.colliding_pairs.size() == pbs.best_node->getCollidingPairs());
        neighbor.sum_of_costs = pbs.best_node->sum_of_costs;
        return true;
    }
    else // stick to old paths
    {
        if (!neighbor.old_paths.empty())
        {
            for (int id : neighbor.agents)
            {
                path_table.insertPath(agents[id].id);
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        num_of_failures++;
        return false;
    }
}
bool SippsSolver::runPP()
{
    auto shuffled_agents = neighbor.agents;
    // std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
    // do not shuffle again
    if (screen >= 2) {
        cout<<"Neighbors_set: ";
        for (auto id : shuffled_agents)
            cout << id << ", ";
        cout << endl;
    }
    int remaining_agents = (int)shuffled_agents.size();
    auto p = shuffled_agents.begin();
    neighbor.sum_of_costs = 0;
    neighbor.colliding_pairs.clear();
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = min(time_limit - runtime, replan_time_limit);
    auto time = Time::now();
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, nullptr, &path_table);
    while (p != shuffled_agents.end() && ((fsec)(Time::now() - time)).count() < T)
    {
        int id = *p;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path);
        assert(agents[id].path_planner->num_collisions > 0 or
            !updateCollidingPairs(neighbor.colliding_pairs, agents[id].id, agents[id].path));
        neighbor.sum_of_costs += (int)agents[id].path.size() - 1;
        remaining_agents--;
        if (screen >= 3)
        {
            runtime = ((fsec)(Time::now() - start_time)).count();
            cout << "After agent " << id << ": Remaining agents = " << remaining_agents <<
                 ", colliding pairs = " << neighbor.colliding_pairs.size() <<
                 ", LL nodes = " << agents[id].path_planner->getNumExpanded() <<
                 ", remaining time = " << time_limit - runtime << " seconds. " << endl;
        }
        if (neighbor.colliding_pairs.size() >= neighbor.old_colliding_pairs.size())
            break;
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
    }
    if (p == shuffled_agents.end() && neighbor.colliding_pairs.size() <= neighbor.old_colliding_pairs.size()) // accept new paths
    {
        return true;
    }
    else // stick to old paths
    {
        if (p != shuffled_agents.end())
            num_of_failures++;
        auto p2 = shuffled_agents.begin();
        while (p2 != p)
        {
            int a = *p2;
            path_table.deletePath(agents[a].id);
            ++p2;
        }
        if (!neighbor.old_paths.empty())
        {
            p2 = neighbor.agents.begin();
            for (int i = 0; i < (int)neighbor.agents.size(); i++)
            {
                int a = *p2;
                agents[a].path = neighbor.old_paths[i];
                path_table.insertPath(agents[a].id);
                ++p2;
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        return false;
    }
}

bool SippsSolver::getInitialSolution(const vector<int>& init_seq)
{
    start_time = Time::now();

    neighbor.agents.clear();
    neighbor.agents.reserve(agents.size());
    sum_of_costs = 0;
    for (int i = 0; i < (int)agents.size(); i++)
    {
        neighbor.agents.push_back(init_seq[i]);
    }

    int remaining_agents = (int)neighbor.agents.size();

    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, nullptr, &path_table);
    set<pair<int, int>> colliding_pairs;
    for (auto id : neighbor.agents)
    {
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        assert(!agents[id].path.empty() && agents[id].path.back().location == agents[id].path_planner->goal_location);
        if (agents[id].path_planner->num_collisions > 0)
            updateCollidingPairs(colliding_pairs, agents[id].id, agents[id].path);
        sum_of_costs += (int)agents[id].path.size() - 1;
        remaining_agents--;
        path_table.insertPath(agents[id].id, agents[id].path);
        runtime = ((fsec)(Time::now() - start_time)).count();
        if (screen >= 3)
        {
            cout << "After agent " << id << ": Remaining agents = " << remaining_agents <<
                 ", colliding pairs = " << colliding_pairs.size() <<
                 ", LL nodes = " << agents[id].path_planner->getNumExpanded() <<
                 ", remaining time = " << time_limit - runtime << " seconds. " << endl;
        }
        if (runtime > time_limit)
            break;
    }

    num_of_colliding_pairs = colliding_pairs.size();
    for(const auto& agent_pair : colliding_pairs)
    {
        collision_graph[agent_pair.first].emplace(agent_pair.second);
        collision_graph[agent_pair.second].emplace(agent_pair.first);
    }
    if (screen >= 2)
        printCollisionGraph();
    return remaining_agents == 0;
}

// return true if the new p[ath has collisions;
bool SippsSolver::updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const
{
    bool succ = false;
    if (path.size() < 2)
        return succ;
    for (int t = 1; t < (int)path.size(); t++)
    {
        int from = path[t - 1].location;
        int to = path[t].location;
        if ((int)path_table.table[to].size() > t) // vertex conflicts
        {
            for (auto id : path_table.table[to][t])
            {
                succ = true;
                colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
            }
        }
        if (from != to && path_table.table[to].size() >= t && path_table.table[from].size() > t) // edge conflicts
        {
            for (auto a1 : path_table.table[to][t - 1])
            {
                for (auto a2: path_table.table[from][t])
                {
                    if (a1 == a2)
                    {
                        succ = true;
                        colliding_pairs.emplace(min(agent_id, a1), max(agent_id, a1));
                        break;
                    }
                }
            }
        }
        //auto id = getAgentWithTarget(to, t);
        //if (id >= 0) // this agent traverses the target of another agent
        //    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        if (!path_table.goals.empty() && path_table.goals[to] < t) // target conflicts
        { // this agent traverses the target of another agent
            for (auto id : path_table.table[to][path_table.goals[to]]) // look at all agents at the goal time
            {
                if (agents[id].path.back().location == to) // if agent id's goal is to, then this is the agent we want
                {
                    succ = true;
                    colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
                    break;
                }
            }
        }
    }
    int goal = path.back().location; // target conflicts - some other agent traverses the target of this agent
    for (int t = (int)path.size(); t < path_table.table[goal].size(); t++)
    {
        for (auto id : path_table.table[goal][t])
        {
            succ = true;
            colliding_pairs.emplace(min(agent_id, id), max(agent_id, id));
        }
    }
    return succ;
}

void SippsSolver::chooseDestroyHeuristicbyALNS()
{
    rouletteWheel();
    switch (selected_neighbor)
    {
        case 0 : init_destroy_strategy = TARGET_BASED; break;
        case 1 : init_destroy_strategy = COLLISION_BASED; break;
        case 2 : init_destroy_strategy = RANDOM_BASED; break;
        default : cerr << "ERROR" << endl; exit(-1);
    }
}

bool SippsSolver::generateNeighborByCollisionGraph()
{
    vector<int> all_vertices;
    all_vertices.reserve(collision_graph.size());
    for (int i = 0; i < (int)collision_graph.size(); i++)
    {
        if (!collision_graph[i].empty())
            all_vertices.push_back(i);
    }
    unordered_map<int, set<int>> G;
    auto v = all_vertices[rand() % all_vertices.size()]; // pick a random vertex
    findConnectedComponent(collision_graph, v, G);
    assert(G.size() > 1);

    assert(neighbor_size <= (int)agents.size());
    set<int> neighbors_set;
    if ((int)G.size() <= neighbor_size)
    {
        for (const auto& node : G)
            neighbors_set.insert(node.first);
        int count = 0;
        while ((int)neighbors_set.size() < neighbor_size && count < 10)
        {
            int a1 = *std::next(neighbors_set.begin(), rand() % neighbors_set.size());
            int a2 = randomWalk(a1);
            if (a2 != NO_AGENT)
                neighbors_set.insert(a2);
            else
                count++;
        }
    }
    else
    {
        int a = std::next(G.begin(), rand() % G.size())->first;
        neighbors_set.insert(a);
        while ((int)neighbors_set.size() < neighbor_size)
        {
            a = *std::next(G[a].begin(), rand() % G[a].size());
            neighbors_set.insert(a);
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by collision graph" << endl;
    return true;

}
bool SippsSolver::generateNeighborByTarget()
{
    int a = -1;
    auto r = rand() % (num_of_colliding_pairs * 2);
    int sum = 0;
    for (int i = 0 ; i < (int)collision_graph.size(); i++)
    {
        sum += (int)collision_graph[i].size();
        if (r <= sum and !collision_graph[i].empty())
        {
            a = i;
            break;
        }
    }
    assert(a != -1 and !collision_graph[a].empty());
    set<pair<int,int>> A_start; // an ordered set of (time, id) pair.
    set<int> A_target;


    for(int t = 0 ;t< path_table.table[agents[a].path_planner->start_location].size();t++){
        for(auto id : path_table.table[agents[a].path_planner->start_location][t]){
            if (id!=a)
                A_start.insert(make_pair(t,id));
        }
    }



    agents[a].path_planner->findMinimumSetofColldingTargets(goal_table,A_target);// generate non-wait path and collect A_target


    if (screen >= 3){
        cout<<"     Selected a : "<< a<<endl;
        cout<<"     Select A_start: ";
        for(auto e: A_start)
            cout<<"("<<e.first<<","<<e.second<<"), ";
        cout<<endl;
        cout<<"     Select A_target: ";
        for(auto e: A_target)
            cout<<e<<", ";
        cout<<endl;
    }

    set<int> neighbors_set;

    neighbors_set.insert(a);

    if(A_start.size() + A_target.size() >= neighbor_size-1){
        if (A_start.empty()){
            vector<int> shuffled_agents;
            shuffled_agents.assign(A_target.begin(),A_target.end());
            std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
            neighbors_set.insert(shuffled_agents.begin(), shuffled_agents.begin() + neighbor_size-1);
        }
        else if (A_target.size() >= neighbor_size){
            vector<int> shuffled_agents;
            shuffled_agents.assign(A_target.begin(),A_target.end());
            std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
            neighbors_set.insert(shuffled_agents.begin(), shuffled_agents.begin() + neighbor_size-2);

            neighbors_set.insert(A_start.begin()->second);
        }
        else{
            neighbors_set.insert(A_target.begin(), A_target.end());
            for(auto e : A_start){
                //A_start is ordered by time.
                if (neighbors_set.size()>= neighbor_size)
                    break;
                neighbors_set.insert(e.second);

            }
        }
    }
    else if (!A_start.empty() || !A_target.empty()){
        neighbors_set.insert(A_target.begin(), A_target.end());
        for(auto e : A_start){
            neighbors_set.insert(e.second);
        }

        set<int> tabu_set;
        while(neighbors_set.size()<neighbor_size){
            int rand_int = rand() % neighbors_set.size();
            auto it = neighbors_set.begin();
            std::advance(it, rand_int);
            a = *it;
            tabu_set.insert(a);

            if(tabu_set.size() == neighbors_set.size())
                break;

            vector<int> targets;
            for(auto p: agents[a].path){
                if(goal_table[p.location]>-1){
                    targets.push_back(goal_table[p.location]);
                }
            }

            if(targets.empty())
                continue;
            rand_int = rand() %targets.size();
            neighbors_set.insert(*(targets.begin()+rand_int));
        }
    }



    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by target" << endl;
    return true;
}
bool SippsSolver::generateNeighborRandomly()
{
    if (neighbor_size >= agents.size())
    {
        neighbor.agents.resize(agents.size());
        for (int i = 0; i < (int)agents.size(); i++)
            neighbor.agents[i] = i;
        return true;
    }
    set<int> neighbors_set;
    auto total = num_of_colliding_pairs * 2 + agents.size();
    while(neighbors_set.size() < neighbor_size)
    {
        vector<int> r(neighbor_size - neighbors_set.size());
        for (auto i = 0; i < neighbor_size - neighbors_set.size(); i++)
            r[i] = rand() % total;
        std::sort(r.begin(), r.end());
        int sum = 0;
        for (int i = 0, j = 0; i < agents.size() and j < r.size(); i++)
        {
            sum += (int)collision_graph[i].size() + 1;
            if (sum >= r[j])
            {
                neighbors_set.insert(i);
                while (j < r.size() and sum >= r[j])
                    j++;
            }
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors randomly" << endl;
    return true;
}

// Random walk; return the first agent that the agent collides with
int SippsSolver::randomWalk(int agent_id)
{
    int t = rand() % agents[agent_id].path.size();
    int loc = agents[agent_id].path[t].location;
    while (t <= path_table.makespan and
           (path_table.table[loc].size() <= t or
           path_table.table[loc][t].empty() or
           (path_table.table[loc][t].size() == 1 and path_table.table[loc][t].front() == agent_id)))
    {
        auto next_locs = instance.getNeighbors(loc);
        next_locs.push_back(loc);
        int step = rand() % next_locs.size();
        auto it = next_locs.begin();
        loc = *std::next(next_locs.begin(), rand() % next_locs.size());
        t = t + 1;
    }
    if (t > path_table.makespan)
        return NO_AGENT;
    else
        return *std::next(path_table.table[loc][t].begin(), rand() % path_table.table[loc][t].size());
}

void SippsSolver::writeIterStatsToFile(const string & file_name) const
{
    std::ofstream output;
    output.open(file_name);
    // header
    output << //"num of agents," <<
           "sum of costs," <<
           "num of colliding pairs," <<
           "runtime" << //"," <<
           //"MAPF algorithm" <<
           endl;

    for (const auto &data : iteration_stats)
    {
        output << //data.num_of_agents << "," <<
               data.sum_of_costs << "," <<
               data.num_of_colliding_pairs << "," <<
               data.runtime << //"," <<
               // data.algorithm <<
               endl;
    }
    output.close();
}

void SippsSolver::writeResultToFile(const string & file_name, int sum_of_distances, double preprocessing_time) const
{
    std::ifstream infile(file_name);
    bool exist = infile.good();
    infile.close();
    if (!exist)
    {
        ofstream addHeads(file_name);
        addHeads << "runtime,num of collisions,solution cost,initial collisions,initial solution cost," <<
                 "sum of distances,iterations,group size," <<
                 "runtime of initial solution,area under curve," <<
                 "LL expanded nodes,LL generated,LL reopened,LL runs," <<
                 "preprocessing runtime,solver name,instance name" << endl;
        addHeads.close();
    }
    uint64_t num_LL_expanded = 0, num_LL_generated = 0, num_LL_reopened = 0, num_LL_runs = 0;
    for (auto & agent : agents)
    {
        agent.path_planner->reset();
        num_LL_expanded += agent.path_planner->accumulated_num_expanded;
        num_LL_generated += agent.path_planner->accumulated_num_generated;
        num_LL_reopened += agent.path_planner->accumulated_num_reopened;
        num_LL_runs += agent.path_planner->num_runs;
    }
    double auc = 0;
    if (!iteration_stats.empty())
    {
        auto prev = iteration_stats.begin();
        auto curr = prev;
        ++curr;
        while (curr != iteration_stats.end() && curr->runtime < time_limit)
        {
            auc += prev->num_of_colliding_pairs * (curr->runtime - prev->runtime);
            prev = curr;
            ++curr;
        }
        auc += prev->num_of_colliding_pairs * (time_limit - prev->runtime);
    }

    ofstream stats(file_name, std::ios::app);
    stats << runtime << "," << iteration_stats.back().num_of_colliding_pairs << "," <<
          sum_of_costs << "," << iteration_stats.front().num_of_colliding_pairs << "," <<
          iteration_stats.front().sum_of_costs << "," << sum_of_distances << "," <<
          iteration_stats.size() << "," << average_group_size << "," <<
          iteration_stats.front().runtime << "," << auc << "," <<
          num_LL_expanded << "," << num_LL_generated << "," << num_LL_reopened << "," << num_LL_runs << "," <<
          preprocessing_time << "," << getSolverName() << "," << instance.getInstanceName() << endl;
    stats.close();
}

void SippsSolver::printCollisionGraph() const
{
    cout << "Collision graph: ";
    int edges = 0;
    for (size_t i = 0; i < collision_graph.size(); i++)
    {
        for (int j : collision_graph[i])
        {
            if (i < j)
            {
                cout << "(" << i << "," << j << "),";
                edges++;
            }
        }
    }
    cout << endl <<  "|V|=" << collision_graph.size() << ", |E|=" << edges << endl;
}


unordered_map<int, set<int>>& SippsSolver::findConnectedComponent(const vector<set<int>>& graph, int vertex,
                                                               unordered_map<int, set<int>>& sub_graph)
{
    std::queue<int> Q;
    Q.push(vertex);
    sub_graph.emplace(vertex, graph[vertex]);
    while (!Q.empty())
    {
        auto v = Q.front(); Q.pop();
        for (const auto & u : graph[v])
        {
            auto ret = sub_graph.emplace(u, graph[u]);
            if (ret.second) // insert successfully
                Q.push(u);
        }
    }
    return sub_graph;
}

void SippsSolver::printPath() const
{
    for (const auto& agent : agents)
        cout << "Agent " << agent.id << ": " << agent.path << endl;
}

void SippsSolver::PrintPriorities() const{
    cout << " Priorities: ";
    for (auto p : priorities){
        cout << p << " > ";
    }
    cout <<endl;
}

void SippsSolver::printResult()
{
    average_group_size = - iteration_stats.front().num_of_agents;
    for (const auto& data : iteration_stats)
        average_group_size += data.num_of_agents;
    if (average_group_size > 0)
        average_group_size /= (double)(iteration_stats.size() - 1);
    assert(!iteration_stats.empty());
    cout << "\t" << getSolverName() << ": "
         << "runtime = " << runtime << ", "
         << "iterations = " << iteration_stats.size() << ", "
         << "colliding pairs = " << num_of_colliding_pairs << ", "
         << "initial colliding pairs = " << iteration_stats.front().num_of_colliding_pairs << ", "
         << "solution cost = " << sum_of_costs << ", "
         << "initial solution cost = " << iteration_stats.front().sum_of_costs << ", "
         << "failed iterations = " << num_of_failures << endl;
}

void SippsSolver::clear()
{
    path_table.clear();
    collision_graph.clear();
    goal_table.clear();
}


bool SippsSolver::validatePathTable() const
{
    for (auto i = 0; i < agents.size(); i++)
        assert(path_table.getPath(i) == &agents[i].path);
    return true;
}


namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;
PYBIND11_MODULE(solver, m) {
    m.doc() = "optional module docstring";

    py::class_<SolverWrapper>(m, "SolverWrapper")
    .def(py::init<py::array_t<double>, int , int ,  
        py::array_t<double>, int, int, int, int, double>())
    .def("calcInitialSolution", &SolverWrapper::calcInitialSolution)
    .def("step", &SolverWrapper::step)
    .def("validateSolutions", &SolverWrapper::validateSolutions)
    .def("getSolutions", &SolverWrapper::getSolutions)
    .def("getPriorities", &SolverWrapper::getPriorities)
    .def("getCollisionEdgeIndex", &SolverWrapper::getCollisionEdgeIndex)
    .def("getTragetEdgeIndex", &SolverWrapper::getTragetEdgeIndex)
    .def("solveSingleAgent", &SolverWrapper::solveSingleAgent)
    .def("repairSolutions", &SolverWrapper::repairSolutions)
    ;
}
