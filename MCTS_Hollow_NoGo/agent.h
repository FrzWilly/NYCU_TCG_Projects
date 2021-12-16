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
// UCB exploration ratio
#define C 1.44
//initial winrate of a unexpanded node in a Monte-Carlo tree
#define INIT_WINRATE 0
//how many simulations have to perform per step
#define SIM_COUNT 100

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
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
	MCTS_agent(const std::string& args = "") : random_agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
		if (meta.find("C") != meta.end())
			exploration_w = double(meta["C"]);
		else
			exploration_w = C;
		if (meta.find("fix_sim") != meta.end())
			sim_count = double(meta["fix_sim"]);
		else
			sim_count = SIM_COUNT;
	}
	virtual ~MCTS_agent() {}

protected:
	std::default_random_engine engine;
	double exploration_w;
	int sim_count;
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
		who(board::empty), oppo(board::empty), MCT(std::make_shared<tree_node>(who, exploration_w), exploration_w), turn(0), won(0), lost(0){
		if (name().find_first_of("[]():; ") != std::string::npos)
			throw std::invalid_argument("invalid name: " + name());
		else{
			strategy = name();
		}
		if (role() == "black") {
			who = board::black;
			oppo = board::white;
		}
		if (role() == "white") {
			who = board::white;
			oppo = board::black;
		}
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		for (size_t i = 0; i < oppo_space.size(); i++)
			oppo_space[i] = action::place(i, oppo);

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

		void visit_record(int result){
			// double old = winrate;
			//std::cout<<"result: "<<result<<std::endl;
			// std::cout<<"\n\n"<<winrate<<" "<<visit_count<<"\n";
			wincount += result;
			// double win = (double)(winrate*visit_count + (double)result);
			visit_count++;
			// std::cout<<win<<" "<<visit_count<<"\n\n";
			// winrate = win / visit_count;
			// if(winrate>1)
				//winrate = old;
			 	// std::cout<<"old winrate: "<<old<<" new winrate: "<<winrate<<std::endl;
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
			double score, best_score = -99999;
			int count, best_count = 0;
			auto iter = children.begin();
			while(iter != children.end()){
				score = UCB_score(iter->first, role);
				count = iter->second->get_count();
				if(count > best_count){
					// std::cout<<score<<" > "<<best_score<<"\n";
					best_count = count;
					best_action = iter->first;
				}
				// std::cout << "[" << iter->first << ","
                //     << iter->second->get_count()<< ","
				// 	<< iter->second->get_winrate()<< ","
				// 	<< UCB_score(iter->first, role) << "]\n";
        		++iter;
			}
			// std::cout<<"best score : "<<best_score<<"\n";
			// std::cout<<"best action : "<<best_action<<"\n";
			return best_action;
		}

		board::piece_type& get_role(){
			return role;
		}

		double UCB_score(action::place move, board::piece_type who){
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
					return 99;
				else
					return 0;
				c_wincount = INIT_WINRATE;
				c_vcount = 1;
			}
			// std::cout<<expw * sqrt(log((double)visit_count)/c_vcount)<<"\n";
			return (double)c_wincount/c_vcount + expw * sqrt(log((double)visit_count)/c_vcount);
		}

		double get_wincount(){
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

		void set_wincount(int value){
			wincount = value;
			visit_count++;
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

		// tree operator =(tree old){
		// 	tree new_tree(old.get_root(), old.get_exp());
		// 	return new_tree;
		// }

		void move_root(action::place mv){
			if(root->has_child(mv)){
				// std::cout<<"move root A\n";
				// root->list_all_children();
				// std::cout<<root->child(mv)->get_winrate();
				// std::cout<<"child exist\n";
				root = root->child(mv);
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
				root = root->child(mv);
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

	virtual action random_take_action(const board& state, board::piece_type role) {
		std::vector<action::place> auto_space(space);
		if(role == oppo){
			std::vector<action::place> newspace(oppo_space);
			auto_space = newspace;
		}
		// std::cout<<"role: "<<role<<std::endl;
		std::shuffle(auto_space.begin(), auto_space.end(), engine);
		
		for (const action::place& move : auto_space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
			// else if(move.apply(after) == board::illegal_turn){
			// 	// std::cout<<"wrong role..."<<std::endl;
			// 	// std::cout<<"move: "<<move<<std::endl;
			// 	return action();
			// }
		}
		// std::cout<<"no random action :(\n";
		// std::cout<<state<<std::endl; mu87;r6nu8		1
		return action();
	}

	virtual int simulation(const board& state, std::shared_ptr<tree_node> node) {
		board after = state;
		board::piece_type role = node->get_role(), op;
		int reward;
		if(role == who){
			reward = 0;
			op = oppo;
		}
		else{
			reward = 1;
			op = who;
		}
		int count = 0;
		// std::cout<<after<<std::endl;
		// std::cout<<"simulation start as: "<<role<<" "<<(role xor 1)<<std::endl;
		while(1){
			count++;
			//s// std::cout<<role<<" "<<(role xor 1)<<std::endl;
			// std::cout<<after<<std::endl;
			action::place move = random_take_action(after, role);
			if(move == action()) {
				// std::cout<<"simulation end as: "<<role<<" total "<<count<<" turns"<<std::endl;
				return reward*5;
			}
			else
				move.apply(after);

			move = random_take_action(after, op);
			if(move == action()) {
				// std::cout<<"simulation end as: "<<op<<" total "<<count<<" turns"<<std::endl;
				// std::cout<<"simulation end as: "<<(role xor 1)<<std::endl;
				return (reward xor 1)*5;
			}
			else
				move.apply(after);
		}
	}

	virtual std::pair<action, int> selection(const board& state, std::shared_ptr<tree_node> node) {
		//std::cout<<"in selection, "
		action::place best_move;
		double best_score = -999999;
		board best_after;
		auto auto_space = space;

		int result;
		if(node->get_role() == oppo)
			auto_space = oppo_space;
		std::shuffle(auto_space.begin(), auto_space.end(), engine);
		for (const action::place& move : auto_space) {
			board after = state;
			//// std::cout<<after<<"\n\n\n"<<std::endl;
			double score;
			// std::cout<<node->get_role()<<" "<<who<<" "<<move<<std::endl;
			if (move.apply(after) == board::legal){
				if(node->get_role() == who)
					score = node->UCB_score(move, who);
				else
					score = -node->UCB_score(move, who);
				// if(score == -999){
				// 	score = 0;
				// }
				if(score > best_score){
					best_move = move;
					best_score = score;
					best_after = after;
				}
			}
			// if(turn > 30 &&best_score != -999999 && best_score != 999 && best_score != 0)
			// 	// std::cout<<"best_score "<<best_score<<std::endl;
			// else if(move.apply(after) == board::illegal_turn){
			// 	std::cout<<"wrong role !!!"<<std::endl;
			// }
				
		}
		if(best_score == -999999){
			// std::cout<<"can't find legal action as child\n";
			// std::cout<<"play as "<<node->get_role()<<", board: \n";
			// std::cout<<state<<std::endl;
			int win = 0;
			if(node->get_role() == oppo)
				win = 999;
			node->set_wincount(win);
			node->set_leaf();
			return std::pair<action, int>(action(), win%2);

		}
		if(node->has_child(best_move)){
			// std::cout<<"select\n";
			std::pair<action, int> back_prop;
			back_prop = selection(best_after, node->child(best_move));
			// if(node->get_role() == oppo){
			// 	std::cout<<"oppo node update\n";
			// }
			node->visit_record(back_prop.second);
			// if(node->get_role() == oppo){
			// 	std::cout<<"oppo node update result: "<<node->get_winrate()<<", "<<node->get_count()<<"\n";
			// }
			// std::cout<<"select return\n";
		}
		else{
			// std::cout<<"expand\n";
			board::piece_type oppo_role;
			if(node->get_role() == board::white)
				oppo_role = board::black;
			else
				oppo_role = board::white;
			node->new_child(oppo_role, best_move);
			result = simulation(best_after, node->child(best_move));
			// if(oppo_role == oppo){
			// 	std::cout<<"oppo node new child update\n";
			// }
			node->child(best_move)->visit_record(result);
			// if(oppo_role == oppo){
			// 	std::cout<<"oppo node update result: "<<node->child(best_move)->get_winrate()<<", "<<node->child(best_move)->get_count()<<"\n";
			// }
			// if(node->get_role() == oppo){
			// 	std::cout<<"oppo node update\n";
			// }
			node->visit_record(result);
			// if(node->get_role() == oppo){
			// 	std::cout<<"oppo node update result: "<<node->get_winrate()<<", "<<node->get_count()<<"\n";
			// }
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
			// std::cout<<"can't find opponent's move"<<std::endl;
			// std::cout<<"last board:\n";
			// std::cout<<last_board<<std::endl;
			// std::cout<<"\n\ncurrent board:\n";
			// std::cout<<state<<std::endl;
			
			/*new game*/
			// std::cout<<"\n\n\n------reset------\n\n\n";
			// if(MCT.get_root()->get_winrate()==1){
			// 	won++;
			// 	std::cout<<"game won, w-l: "<<won<<" : "<<lost<<"\n";
			// }
			// else if(MCT.get_root()->get_winrate()<=0){
			// 	lost++;
			// 	std::cout<<"game lost, w-l: "<<won<<" : "<<lost<<"\n";
			// }
			// else{
			// 	std::cout<<"game finished with root winrate "<<MCT.get_root()->get_winrate()<<"\n";
			// }
			MCT.reset_tree(who);
			turn = 0;
			// std::cout<<"reset\n";
		}
		else{
			// std::cout<<"check opponent root\n";
			// if(MCT.get_root()->has_child(oppo_mv))
			// 	MCT.get_root()->child(oppo_mv)->list_all_children();
			// MCT.get_root()->list_all_children();
			MCT.move_root(oppo_mv);
		}
	}

	action early(std::shared_ptr<tree_node> node){
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
		// std::cout<<"child count: "<<count<<std::endl;
		// std::cout<<"most: "<<most<<", second: "<<second<<std::endl;
		if(most - (SIM_COUNT)*0.8 >= second){
			return most_a;
		}
		return action();
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
		//// std::cout<<"take action\n";
		action move;
		//update opponent move if state is not empty board
		if(turn){
			// play as black and not init
			//// std::cout<<"update oppo\n";
			board after = state;
			handle_oppo_turn(after);
		}
		else{
			// play as black init
			// std::cout<<"\n\n\n------reset------\n\n\n";
			// if(MCT.get_root()->get_winrate()==1){
			// 	won++;
			// 	std::cout<<"game won, w-l: "<<won<<" : "<<lost<<"\n";
			// }
			// else if(MCT.get_root()->get_winrate()<=0){
			// 	lost++;
			// 	std::cout<<"game lost, w-l: "<<won<<" : "<<lost<<"\n";
			// }
			// else{
			// 	std::cout<<"game finished with root winrate "<<MCT.get_root()->get_winrate()<<"\n";
			// }
			MCT.reset_tree(who);
			turn = 0;
			// std::cout<<"reset\n";
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
		// if(MCT.get_root()->get_winrate()){
		// 	std::cout<<state<<std::endl;
		// 	std::cout<<"root visit count: "<<MCT.get_root()->get_count()<<std::endl;
		// 	std::cout<<"root winrate: "<<MCT.get_root()->get_winrate()<<std::endl;
		// 	if(MCT.get_root()->UCB_score(move, who)!=999)
		// 		std::cout<<"move UCB score: "<<MCT.get_root()->UCB_score(move, who)<<std::endl;
		// }
		action most_visited = action();
		for (int i=0;i<sim_count;i++) {
			if(MCT.get_root()->check_leaf()){
				// std::cout<<"root at leaf"<<std::endl;
				move = action();
				break;
			}
			// std::cout<<"root visit_count: "<<MCT.get_root()->get_count()<<std::endl;
			board after = state;
			move = selection(after, MCT.get_root()).first;
			/* early */
			// most_visited = early(MCT.get_root());
			// if(most_visited == action())
			// 	move = selection(after, MCT.get_root()).first;
			// else{
			// 	// std::cout<<"early activated"<<std::endl;
			// 	move = most_visited;
			// 	break;
			// }

			// if(MCT.get_root()->child(move)->check_leaf() && MCT.get_root()->child(move)->get_winrate()==1){
			// 	// std::cout<<"found win\n";
			// 	break;
			// }
			// std::cout<<"chosen move: "<<move<<std::endl;
			// std::cout<<"simulation count: "<<i<<std::endl;
			// std::cout<<"root winrate: "<<MCT.get_root()->get_winrate()<<std::endl;
			// std::cout<<"move UCB score: "<<MCT.get_root()->UCB_score(move)<<std::endl;
		}
		move = MCT.get_root()->best_children();
		// std::cout<<"\n\n";
		// std::cout<<"after simulation childrean:\n";
		// MCT.get_root()->list_all_children();
		// std::cout<<"choose move: "<<move<<"\n";
		// std::cout<<"\nafter simulation new root childrean:\n";
		// if(MCT.get_root()->has_child(move))
		// 	MCT.get_root()->child(move)->list_all_children();
		// else{
		// 	std::cout<<"None\n";
		// }
		// std::cout<<"\n\n";
		// if(most_visited == action()){
		// 	move = MCT.get_root()->best_children();
		// }
		// if(MCT.get_root()->get_winrate()){
		// 	std::cout<<"root visit count: "<<MCT.get_root()->get_count()<<std::endl;
		// 	std::cout<<"root winrate: "<<MCT.get_root()->get_winrate()<<std::endl;
		// 	if(MCT.get_root()->UCB_score(move, who)!=999)
		// 		std::cout<<"move UCB score: "<<MCT.get_root()->UCB_score(move, who)<<std::endl;
		// 	MCT.get_root()->list_all_children();
		// }
		// if(MCT.get_root()->UCB_score(move) == 100){
		// 	won++;
		// 	// std::cout<<"game won, w-l: "<<won<<" : "<<lost<<"\n";
		// }
		// else if(MCT.get_root()->UCB_score(move) < 0){
		// 	lost++;
		// 	// std::cout<<"game lost, w-l: "<<won<<" : "<<lost<<"\n";
		// }
		
		board after = state;
		MCT.move_root(move);
		
		// std::cout<<"turn "<<turn<<" ended"<<std::endl;
		if(move.apply(after) == board::legal){
			last_board = after;
			// if(MCT.get_root()->get_winrate()<0.5){
			// 	std::cout<<"move: "<<move<<std::endl;
			// 	std::cout<<"root role: "<<MCT.get_root()->get_role()<<std::endl;
			// 	std::cout<<after<<std::endl;
			// 	std::cout<<"\n\n---------------------\n\n"<<std::endl;
			// }
			return move;
		}
		else
			return action();
	}

	virtual action take_action(const board& state) {
		if(strategy == "mcts")
			return mcts_take_action(state);
		else if(strategy == "random")
			return random_player_take_action(state);

		return action();
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
	int won;
	int lost;
};


