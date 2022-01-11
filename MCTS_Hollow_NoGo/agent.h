/**
 * Framework for NoGo and similar games (C++ 11)
 * agent.h: Define the behavior of variants of the player
 *
 * Author: Theory of Computer Games (TCG 2021)
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <cstring>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include <fstream>
#include <memory>
#include <ctime>
#include <unistd.h>
#include <thread>
#include <queue>
#include <algorithm>
#include <mutex>

// UCB exploration ratio
#define C 1.44
//initial winrate of a unexpanded node in a Monte-Carlo tree
#define INIT_WINRATE 0
//how many simulations have to perform per step
#define SIM_COUNT 100
//win value weight
#define WIN_WEIGHT 2
//default basic formula constant for time management
#define BASIC_C 30
//default enhanced formula parameter max_ply for time management
#define ENHANCED_PEAK 15
//initial time limit(ms), less than actual time limit just in case
#define INIT_TIME 300.0
//early activate threshold
#define EARLY_T 5000
// equally expand thinking time to fully utilize given time if 
// some time-management use only part of given time
// set this to 1 to turn-off bonus
#define TIME_BONUS 1

std::mutex mu;

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown search=unknown" + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }
	virtual std::string search() const { return property("search"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents using MCTS method
 */
class MCTS_agent : public random_agent {
public:
	MCTS_agent(const std::string& args = "") : random_agent(args),
		basic_const(0), enhanced_peak(0), use_time_management(false),
		 unst_N(0), time_bonus(1), leaf_parallel(0), earlyc_p(0){
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
		if (meta.find("C") != meta.end())
			exploration_w = double(meta["C"]);
		else
			exploration_w = C;

		if (meta.find("fix_sim") != meta.end())
			sim_count = int(meta["fix_sim"]);
		else
			sim_count = SIM_COUNT;

		if (meta.find("enhanced_f") != meta.end()){
			enhanced_peak = int(meta["enhanced_f"]);
			sim_count = 99999999;
			use_time_management = true;
		}
		if(meta.find("basic_f") != meta.end()){
			basic_const = int(meta["basic_f"]);
			sim_count = 99999999;
			use_time_management = true;
		}
		else
			basic_const = BASIC_C;

		if (meta.find("early") != meta.end())
			if_early = true;
		else
			if_early = false;

		if (meta.find("early_c") != meta.end()){
			if_early = true;
			earlyc_p = double(meta["early_c"]);
		}
		else
			if_early = false;

		if (meta.find("unst") != meta.end())
			unst_N = int(meta["unst"]);

		if (meta.find("t_bonus") != meta.end())
			time_bonus = double(meta["t_bonus"]);

		if (meta.find("p_leaf") != meta.end())
			leaf_parallel = int(meta["p_leaf"]);

		// std::cout<<"search: "<<search<<std::endl;
	}
	virtual ~MCTS_agent() {}

protected:
	std::default_random_engine engine;
	double exploration_w;
	int basic_const;
	int enhanced_peak;
	int sim_count;
	bool if_early;
	bool use_time_management;
	int unst_N;
	double time_bonus;
	int leaf_parallel;
	double earlyc_p;
	// std::string search;
};

/**
 * random player for both side
 * put a legal piece randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), who(board::empty) {
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") who = board::black;
		if (role() == "white") who = board::white;
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
	}

	virtual action take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

private:
	std::vector<action::place> space;
	board::piece_type who;
};

/**
 * MCTS player for both side
 */
class MCTS_player : public MCTS_agent {
public:
	MCTS_player(const std::string& args = "") : MCTS_agent("name=unknown role=unknown " + args),
		space(board::size_x * board::size_y), oppo_space(board::size_x * board::size_y),
		who(board::empty), oppo(board::empty), MCT(std::make_shared<tree_node>(who, exploration_w), exploration_w),
		 turn(0), remaining_time(INIT_TIME){
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		if (role() == "black") {
			who = board::black;
			oppo = board::white;
		}
		if (role() == "white") {
			who = board::white;
			oppo = board::black;
		}
		if(search() !=""){
			strategy = search();
		}
		else
			strategy = "random";
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		for (size_t i = 0; i < oppo_space.size(); i++)
			oppo_space[i] = action::place(i, oppo);

		if(leaf_parallel)
			threads.resize(leaf_parallel);

		MCT = tree(std::make_shared<tree_node>(who, exploration_w), exploration_w);
	}

	class tree_node
	: public std::enable_shared_from_this<tree_node>
	{
	public:
		tree_node(board::piece_type role, action::place mv, double exp_w) :
			role(role), move(mv), is_leaf(false), expw(exp_w){
				wincount = INIT_WINRATE;
				visit_count = 0;
			}
		// constructor used for initializing root
		tree_node(board::piece_type role, double exp_w) :
			role(role), move(action()), is_leaf(false), expw(exp_w){
				wincount = INIT_WINRATE;
				visit_count = 0;
		}

		std::shared_ptr<tree_node> get_ptr(){
			return shared_from_this();
		}

		bool has_child(action::place move){
			return children.count(move);
		}

		std::shared_ptr<tree_node> child(action::place move){
			return children[move]->get_ptr();
		}

		void new_child(board::piece_type role, action::place move){
			children[move] = std::make_shared<tree_node>(role, move, expw);
		}

		void visit_record(int result, int leaf_parallel){
			wincount += result;
			visit_count += std::max(1, leaf_parallel);
		}

		void list_all_children(){
			auto iter = children.begin();
			while(iter != children.end()){
				std::cout << "[" << iter->first << ","
                    << iter->second->get_count()<< ","
					<< iter->second->get_wincount()<< ","
					<< UCB_score(iter->first, role) << "]\n";
        		++iter;
			}
		}

		action best_children(){
			action::place best_action = action();
			// double score, best_score = -99999;
			int count, best_count = 0;
			auto iter = children.begin();
			while(iter != children.end()){
				// score = UCB_score(iter->first, role);
				count = iter->second->get_count();
				if(count >= best_count){
					// std::cout<<score<<" > "<<best_score<<"\n";
					// std::cout<<count<<" > "<<best_count<<"\n";
					best_count = count;
					best_action = iter->first;
				}
				// std::cout << "[" << iter->first << ","
                //     << iter->second->get_count()<< ","
				// 	<< iter->second->get_winrate()<< ","
				// 	<< UCB_score(iter->first, role) << "]\n";
        		++iter;
			}
			// std::cout << "best action: " << best_action
			// << "best count" << best_count << "\n";
			return best_action;
		}

		action highest_win_children(){
			action::place best_action = action();
			// double score, best_score = -99999;
			double winrate, best_winrate = 0;
			auto iter = children.begin();
			while(iter != children.end()){
				// score = UCB_score(iter->first, role);
				winrate = (double)(iter->second->get_wincount()) / (iter->second->get_count());
				
				if(winrate >= best_winrate){
					// std::cout<<score<<" > "<<best_score<<"\n";
					best_winrate = winrate;
					best_action = iter->first;
				}
				// std::cout << "[" << iter->first << ","
                //     << iter->second->get_count()<< ","
				// 	<< iter->second->get_winrate()<< ","
				// 	<< UCB_score(iter->first, role) << "]\n";
        		++iter;
			}
			return best_action;
		}

		board::piece_type& get_role(){
			return role;
		}

		double UCB_score(action::place move, board::piece_type who, int leaf_parallel = 0){
			int c_wincount, c_vcount;
			if(has_child(move)){
				std::shared_ptr<tree_node> chld(children[move]->get_ptr());
				if(chld->check_leaf())
					return chld->get_wincount()*200;
				c_wincount = chld->get_wincount();
				c_vcount = chld->get_count();
			}
			else{
				//return 999;
				if(role == who)
					return 999;
				else
					return 0;
				c_vcount = std::max(1, leaf_parallel);
				c_wincount = INIT_WINRATE * c_vcount;
			}
			int p = role == who ? 1 : -1;
			// std::cout<<expw * sqrt(log((double)visit_count)/c_vcount)<<"\n";
			return (double)(p * c_wincount)/c_vcount + expw * sqrt(log((double)visit_count)/c_vcount);
		}

		int get_wincount(){
			return wincount;
		}
		int get_count(){
			return visit_count;
		}

		void set_leaf(){
			is_leaf = true;
		}
		bool check_leaf(){
			return is_leaf;
		}

		void set_wincount(int value, int leaf_parallel){
			wincount = value;
			visit_count += std::max(1, leaf_parallel);
		}
		
	private:
		//int depth;
		board::piece_type role;
		action::place move;
		int wincount;
		int visit_count;
		std::map<action::place, std::shared_ptr<MCTS_player::tree_node> > children;
		bool is_leaf;
		double expw;
		//tree_node* prev;
	};

	class tree{
	public:
		tree(std::shared_ptr<tree_node> rot, double exp_w):
		root(rot->get_ptr()), expw(exp_w){};

		std::shared_ptr<tree_node> get_root(){
			return root->get_ptr();
		}

		void move_root(action::place mv){
			if(root->has_child(mv)){
				// std::cout<<"move root A\n";
				// root->list_all_children();
				// std::cout<<root->child(mv)->get_winrate();
				// std::cout<<"child exist\n";
				root = root->child(mv)->get_ptr();
			}
			else{
				// std::cout<<"move root B, ba ka na !!!\n";
				// std::cout<<"move: "<<mv<<"\n";
				// root->list_all_children();
				board::piece_type oppo;
				if(root->get_role() == board::white)
					oppo = board::black;
				else
					oppo = board::white;
				root->new_child(oppo, mv);
				root = root->child(mv)->get_ptr();
			}
		}

		void reset_tree(board::piece_type who){
			root = std::make_shared<tree_node>(who, expw);
		}
		double get_exp(){
			return expw;
		}
	private:
		std::shared_ptr<tree_node> root;
		double expw;
	};

	// virtual action random_take_action(const board& state, std::vector<action::place>& space, board::piece_type role) {
	// 	std::vector<action::place> *auto_space = &space;
	// 	if(role == oppo){
	// 		auto_space = &oppo_space;
	// 	}
		
	// 	for (const action::place& move : *auto_space) {
	// 		board after = state;
	// 		if (move.apply(after) == board::legal)
	// 			return move;
	// 	}
	// 	return action();
	// }

	int rollout(const board& state) {
			std::vector<board::point> empty;
			for (unsigned i = 0; i < 81; i++)
				if (state(i) == board::empty) empty.push_back(i);
			
			std::shuffle(empty.begin(), empty.end(), engine);
			board rollout = state;
			for (size_t i = 0, n = empty.size(); n;) {
				if (rollout.place(empty[n - 1]) == board::legal) {
					n -= 1;
					i = 0;
				} else if (i < n) {
					std::swap(empty[i++], empty[n - 1]);
				} else {
					n = 0;
				}
			}
			// for (size_t i = 0; i < empty.size(); i++) {
			// 	assert(rollout.place(empty[i]) != board::legal);
			// }
			int outcome = static_cast<board::piece_type>(3 - rollout.info().who_take_turns) == who ? WIN_WEIGHT : 0;
			mu.lock();
			if(leaf_parallel)
				simulation_results.push(outcome);
			mu.unlock();
			return outcome;
	}

	virtual std::pair<action, int> selection(const board& state, std::shared_ptr<tree_node> node) {
		//std::cout<<"in selection, "
		action::place best_move;
		double best_score = -999999;
		board best_after;
		std::vector<action::place> *auto_space = &space;

		int result = 0;
		if(node->get_role() == oppo)
			auto_space = &oppo_space;
		for (const action::place& move : *auto_space) {
			board after = state;
			double score;
			if (move.apply(after) == board::legal){
				score = node->UCB_score(move, who);
				// std::cout<<score<<std::endl;

				if(score > best_score){
					best_move = move;
					best_score = score;
					best_after = after;
				}
			}
		}
		// std::cout<<"----------------------------------\n";
		if(best_score == -999999){
			// std::cout<<"what\n";
			int win = 0;
			if(node->get_role() == oppo)
				win = (node->get_count()+std::max(1,leaf_parallel))*WIN_WEIGHT;
			node->set_wincount(win, leaf_parallel);
			node->set_leaf();
			if(win)win=WIN_WEIGHT;
			return std::pair<action, int>(action(), win);

		}
		if(node->has_child(best_move)){
			std::pair<action, int> back_prop;
			back_prop = selection(best_after, node->child(best_move));
			result = back_prop.second;
			node->visit_record(result, leaf_parallel);
		}
		else{
			// clock_t begin, finish;
			// std::cout<<"expand\n";
			board::piece_type oppo_role;
			if(node->get_role() == board::white)
				oppo_role = board::black;
			else
				oppo_role = board::white;
			// begin = millisec();
			node->new_child(oppo_role, best_move);
			if(leaf_parallel){
				// std::cout<<"here;)\n";
				for (int thread_c = 0; thread_c < leaf_parallel; thread_c++) {
					// std::cout<<"here;) "<<thread_c<<"\n";
					// threads[thread_c] = std::thread(&MCTS_player::simulation, this, best_after, node->child(best_move)->get_ptr());
					threads[thread_c] = std::thread(&MCTS_player::rollout, this, best_after);
				}
				// std::cout<<"-----------------\n";
				for (int thread_c = 0; thread_c < leaf_parallel; thread_c++) {
					threads[thread_c].join();
				}

				while(!simulation_results.empty()){
					result += simulation_results.front();
					simulation_results.pop();
				}
			}
			else
				// result = simulation(best_after, node->child(best_move)->get_ptr());
				result = rollout(best_after);
			// finish = millisec();

			// std::cout<<"leaf parallel: "<<leaf_parallel<<std::endl;
			// std::cout<<"simulation cost time: "<<double(finish - begin)/CLOCKS_PER_SEC<<"\n";
			// if(leaf_parallel)
			// 	std::cout<<"result: "<<result<<"\n";

			node->child(best_move)->visit_record(result, leaf_parallel);
			node->visit_record(result, leaf_parallel);
		}
		return std::pair<action, int>(best_move, result);
	}

	void handle_oppo_turn(const board& state){
		action oppo_mv = action();
		for (const action::place& mv : oppo_space) {
			board after = last_board;
			if (mv.apply(after) == board::legal){
				if(after == state){
					// std::cout<<"find oppo move"<<mv<<std::endl;
					oppo_mv = mv;
					break;
				}
			}
		}
		if(oppo_mv == action()){

			/* can't find opponent's move */
			/* new game */

			// std::cout<<state<<std::endl;
			// std::cout<<last_board<<std::endl;
			std::cout<<"game reset, remain time:"<<remaining_time<<std::endl;

			MCT.reset_tree(who);
			turn = 0;
			// std::cout<<"total cost time: "<<INIT_TIME - remaining_time<<std::endl;
			remaining_time = INIT_TIME;
		}
		else{
			MCT.move_root(oppo_mv);
		}
	}

	action early(std::shared_ptr<tree_node> node, double thinking_time=0){
		int most=0, second=0;
		action most_a, second_a;
		int count=0;
		for (const action::place& move : space) {
			if(node->has_child(move)){
				count++;
				if(node->child(move)->get_count() > most){
					second = most;
					second_a = most_a;
					most = node->child(move)->get_count();
					most_a = move;
				}
				else if(node->child(move)->get_count() > second){
					second = node->child(move)->get_count();
					second_a = move;
				}
			}
		}
		if(earlyc_p==0){
			if(most - EARLY_T*std::max(1, leaf_parallel) >= second){
				return most_a;
			}
		}
		else{
			if(most - thinking_time * simcount_lastturn * earlyc_p * std::max(1, leaf_parallel) >= second){
				return most_a;
			}
		}
		return action();
	}

	bool is_unstable(action &move){
		action winrate_move = MCT.get_root()->highest_win_children();
		move = MCT.get_root()->best_children();
		return (winrate_move != move && move != action());
	}

	virtual action random_player_take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

	virtual action mcts_take_action(const board& state) {
		clock_t start, end;
		start = millisec();
		double thinking_time = 0;
		
		if(enhanced_peak){
			thinking_time = remaining_time / (basic_const + std::max(enhanced_peak - turn*2, 0));
		}
		else if(basic_const){
			thinking_time = remaining_time / basic_const;
		}
		thinking_time *= time_bonus;
		// std::cout<<"thinking_time this step: "<<thinking_time<<std::endl;
		action move;
		//update opponent move if state is not empty board
		if(turn){
			// not init
			handle_oppo_turn(state);
		}
		else{
			// init
			// std::cout<<state<<std::endl;
			std::cout<<"game reset, remain time:"<<remaining_time<<std::endl;

			MCT.reset_tree(who);
			turn = 0;
			remaining_time = INIT_TIME;
		}
		turn++;
		// check root role
		// std::cout<<"root role: "<<MCT.get_root()->get_role()<<std::endl;
		if(MCT.get_root()->get_role() != who){
			std::cout<<"wrong role at root"<<std::endl;
			std::cout<<"\n\ncurrent board:\n";
			std::cout<<state<<std::endl;
			exit(-1);
		}
		action most_visited = action();
		if(if_early){
			most_visited = early(MCT.get_root(), thinking_time);
			/* early: if a node has majority votes just choose it */
			if(most_visited != action()){
				goto selection_end;
			}
		}

		for (int i=0;i<sim_count;i++) {
			end = millisec();
			double cost = (double)(end - start) / CLOCKS_PER_SEC;
			// std::cout<<cost<<" : "<<thinking_time<<std::endl;
			if((cost >= thinking_time) && use_time_management){
				simcount_lastturn = (double)i / thinking_time;
				std::cout<<"leaf parallelization: "<<leaf_parallel<<"\n";
				std::cout<<"rollout count: "<<i * std::max(1, leaf_parallel)<<"\n";
				std::cout<<"avg count per second: "<<(double)i / thinking_time;
				std::cout<<"\n--------------\n\n";
				break;
			}
			if(MCT.get_root()->check_leaf()){
				move = action();
				break;
			}
			board after = state;
			move = selection(after, MCT.get_root()->get_ptr()).first;
			/* check early multiple times version */
			if(earlyc_p && turn > 2){
				most_visited = early(MCT.get_root(), thinking_time - cost);
				if(most_visited != action())
					break;
			}
			// else
			// 	move = selection(after, MCT.get_root()->get_ptr()).first;
		}
		selection_end:
		/* unstable time-management method */
		if(unst_N){
			clock_t start2;
			move = MCT.get_root()->best_children();
			int unst_N_thisstep = unst_N;
			while(unst_N_thisstep-- && is_unstable(move)){
				start2 = millisec();
				for (int i=0;i<sim_count;i++) {
					end = millisec();
					double cost = (double)(end - start2) / CLOCKS_PER_SEC;
					if((cost >= thinking_time/2) && use_time_management){
						simcount_lastturn += i;
						std::cout<<"rollout count: "<<i * std::max(1, leaf_parallel)<<"\n";
						std::cout<<"avg count per second: "<<(double)i / thinking_time;
						std::cout<<"\n--------------\n\n";
						break;
					}
					if(MCT.get_root()->check_leaf()){
						move = action();
						break;
					}
					board after = state;
					move = selection(after, MCT.get_root()->get_ptr()).first;
				}
			}
			move = MCT.get_root()->best_children();
		}
		else
			move = MCT.get_root()->best_children();
			// std::cout<<"----------------------\n";

		// if(leaf_parallel)
		// 	MCT.get_root()->list_all_children();
		
		board after = state;
		MCT.move_root(move);

		// std::cout<<"move: "<<move<<std::endl;
		
		// std::cout<<"turn "<<turn<<" ended"<<std::endl;
		if(move.apply(after) == board::legal){
			last_board = after;
			end = millisec();
			double cost = (double)(end - start) / CLOCKS_PER_SEC;
			remaining_time -= cost;
			return move;
		}
		else
			return action();
	}

	virtual action take_action(const board& state) {
		if(strategy == "MCTS")
			return mcts_take_action(state);
		else if(strategy == "random")
			return random_player_take_action(state);

		return action();
	}

protected:
	static time_t millisec() {
		auto now = std::chrono::system_clock::now().time_since_epoch();
		return std::chrono::duration_cast<std::chrono::milliseconds>(now).count()*1000;
	}

private:
	std::string strategy;
	std::vector<action::place> space;
	std::vector<action::place> oppo_space;
	board::piece_type who;
	board::piece_type oppo;
	tree MCT;
	board last_board;
	int turn;
	double remaining_time;
	std::vector<std::thread> threads;
	std::queue<int> simulation_results;
	int simcount_lastturn;
	// int won;
	// int lost;
};


