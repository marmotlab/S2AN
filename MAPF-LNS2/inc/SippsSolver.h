#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include "BasicLNS.h"

namespace py = pybind11;


enum init_destroy_heuristic { TARGET_BASED, COLLISION_BASED, RANDOM_BASED, INIT_COUNT };


class SippsSolver : public BasicLNS
{
public:
    vector<Agent>& agents;
    int num_of_colliding_pairs = 0;

    SippsSolver(const Instance& instance, vector<Agent>& agents, double time_limit,
            const string & replan_algo_name, const string & init_destory_name, 
            int neighbor_size, int max_num_iter, int screen);

    bool getInitialSolution(const vector<int>& init_seq);
    void writeIterStatsToFile(const string & file_name) const;
    void writeResultToFile(const string & file_name, int sum_of_distances, double preprocessing_time) const;
    string getSolverName() const override { return "SippsSolver(" + replan_algo_name + ")"; }

    void printPath() const;
    void PrintPriorities() const;
    void printResult();
    void clear(); // delete useless data to save memory

    // add interface
    bool step(const vector<int>& replan_seq);
    Eigen::VectorXi getPriorities() const;
    Eigen::MatrixXi getCollisionEdgeIndex() const;
	Eigen::MatrixXi getTragetEdgeIndex() const;

	Eigen::MatrixXi getSolutions() const;
	bool validateSolutions(py::array_t<int> paths_np, int solution_soc) const;
    bool solveSingleAgent(int id, vector<int>& higher_agnets) ;
    int repairSolutions() ;

private:
    string replan_algo_name;
    init_destroy_heuristic init_destroy_strategy = COLLISION_BASED;

    PathTableWC path_table; // 1. stores the paths of all agents in a time-space table;
    // 2. avoid making copies of this variable as much as possible.

    vector<set<int>> collision_graph;
    vector<int> goal_table;
    int num_of_agents;

    // add interface
    // len: num_agents. true if its path is in the path_table.
    vector<bool> path_table_agents;
    vector<int> priorities;  // present priorities. 
    // minial collisions a *. collision pair. 
    vector<pair<int, int>> target_edge_index;  
    int max_num_iter;

    void updatePriorities(const vector<int>& replan_seq);

    bool runPP();
    bool runGCBS();
    bool runPBS();

    bool updateCollidingPairs(set<pair<int, int>>& colliding_pairs, int agent_id, const Path& path) const;

    void chooseDestroyHeuristicbyALNS();

    bool generateNeighborByCollisionGraph();
    bool generateNeighborByTarget();
    bool generateNeighborRandomly();

    // int findRandomAgent() const;
    int randomWalk(int agent_id);

    void printCollisionGraph() const;

    static unordered_map<int, set<int>>& findConnectedComponent(const vector<set<int>>& graph, int vertex,
            unordered_map<int, set<int>>& sub_graph);

    bool validatePathTable() const;
};


class SolverWrapper{
public:
    // TODO argument by reference or by value?
    SolverWrapper(    
        py::array_t<double> world, int world_len, int world_wid, 
        py::array_t<double> goals, int num_agents,
        int max_num_iter, int neighbor_size, int screen,
        double time_limit);
    
    void step(py::array_t<double> replan_seq, int num_seq);
    Eigen::VectorXi getPriorities() const { return sipps_solver->getPriorities();};
	Eigen::MatrixXi getSolutions() const{return sipps_solver->getSolutions();};
	Eigen::MatrixXi getCollisionEdgeIndex() const{return sipps_solver->getCollisionEdgeIndex();};
	Eigen::MatrixXi getTragetEdgeIndex() const{return sipps_solver->getTragetEdgeIndex();};

	bool validateSolutions(py::array_t<int> paths, int solution_soc) const{
        return sipps_solver->validateSolutions(paths, solution_soc);
    };
    bool solveSingleAgent(int id, py::list higher_agnets) {
        auto higher_agnets_vec = higher_agnets.cast<std::vector<int>>();
        return sipps_solver->solveSingleAgent(id, higher_agnets_vec);
    };
    int repairSolutions() {return sipps_solver->repairSolutions(); };
    // abandom usage.
    void calcInitialSolution(py::array_t<double> init_seq, int num_agents);

    // interface to SippsSolver
    vector<Agent> agents;
    SippsSolver* sipps_solver=nullptr;

	// add interface to python
    vector<int> priorities;
	vector<int> m_world;
    vector<int> m_goals;
    int m_world_len, m_world_wid;
	int num_agents;

    vector<int> m_priorities;
    // vector<vector<int>> m_solutions;
    vector<int> m_solutions;
    
    vector<bool> obstacle_map;  // len: map_wid*map_height. true if is obstacle.
    vector<int> start_locations;
    vector<int> goal_locations;
    Instance instance;

	// void Initialize(const Instance& instance, bool sipp, int screen);
    // bool solve(int child_id, PBSNode* parent, int low, int high);
};