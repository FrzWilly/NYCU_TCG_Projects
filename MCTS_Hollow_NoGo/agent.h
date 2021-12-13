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
#define C 1.0
//initial winrate of a unexpanded node in a Monte-Carlo tree
#define INIT_WINRATE 0.5

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
class MCTS_player : public random_agent {
public:
	MCTS_player(const std::string& args = "") : random_agent("name=random role=unknown " + args),
		space(board::size_x * board::size_y), oppo_space(board::size_x * board::size_y),
		who(board::empty), oppo(board::empty), MCT(std::make_shared<tree_node>(who)), turn(0){
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
		if (who == board::empty)
			throw std::invalid_argument("invalid role: " + role());
		for (size_t i = 0; i < space.size(); i++)
			space[i] = action::place(i, who);
		for (size_t i = 0; i < oppo_space.size(); i++)
			oppo_space[i] = action::place(i, oppo);

		MCT = std::make_shared<tree_node>((who));
	}

	class tree_node
	: public std::enable_shared_from_this<tree_node>
	{
	public:
		tree_node(board::piece_type role, action::place mv) :
			role(role), move(mv){
				winrate = INIT_WINRATE;
				visit_count = 0;
			}
		// constructor used for initializing root
		tree_node(board::piece_type role) :
			role(role), move(action()){
				winrate = INIT_WINRATE;
				visit_count = 0;
		}

		// tree_node(tree_node& node) :
		// 	role(node.role), move(node.move){
		// 		winrate = node.winrate;
		// 		visit_count = node.visit_count;
		// }

		std::shared_ptr<tree_node> get_ptr(){
			return shared_from_this();
		}

		bool has_child(action::place move){
			return children.count(move);
		}

		std::shared_ptr<tree_node> child(action::place& move){
			return children[move]->get_ptr();
		}

		void new_child(board::piece_type role, action::place move){
			children[move] = std::make_shared<tree_node>(role, move);
		}

		void visit_record(int result){
			double old = winrate;
			if(result){
				//std::cout<<"result: "<<result<<std::endl;
				double win = winrate*visit_count + result;
				winrate = win / ++visit_count;
				std::cout<<"old winrate: "<<old<<" new winrate: "<<winrate<<std::endl;
			}
			else
				visit_count++;
		}

		board::piece_type& get_role(){
			return role;
		}

		double UCB_score(action::place move){
			double c_winrate, c_vcount;
			if(has_child(move)){
				std::shared_ptr<tree_node> chld = children[move]->get_ptr();
				c_winrate = chld->winrate;
				c_vcount = chld->visit_count;
			}
			else{
				//return 999;
				c_winrate = INIT_WINRATE;
				c_vcount = 0.1;
			}
			return c_winrate + C * sqrt(log(visit_count)/c_vcount);
		}

		double get_winrate(){
			return winrate;
		}
		int get_count(){
			return visit_count;
		}
		
	private:
		//int depth;
		board::piece_type role;
		action::place move;
		double winrate;
		int visit_count;
		std::map<action::place, std::shared_ptr<MCTS_player::tree_node> > children;
		//tree_node* prev;
	};

	class tree{
	public:
		tree(std::shared_ptr<tree_node> root):
		root(root->get_ptr()){};

		std::shared_ptr<tree_node> get_root(){
			return root->get_ptr();
		}

		void move_root(action::place mv){
			if(root->has_child(mv)){
				// std::cout<<"move root A\n";
				root = root->child(mv);
			}
			else{
				// std::cout<<"move root B\n";
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
			root = std::make_shared<tree_node>(who);
		}
	private:
		std::shared_ptr<tree_node> root;
	};

	virtual action random_take_action(const board& state, int role) {
		auto auto_space = space;
		if(role)
			auto_space = oppo_space;
		// std::cout<<"role: "<<role<<std::endl;
		std::shuffle(auto_space.begin(), auto_space.end(), engine);
		
		for (const action::place& move : auto_space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
			else if(move.apply(after) == board::illegal_turn){
				std::cout<<"wrong role..."<<std::endl;
				return action();
			}
		}
		// std::cout<<"no random action :(\n";
		// std::cout<<state<<std::endl;
		return action();
	}

	virtual int simulation(const board& state, std::shared_ptr<tree_node> node) {
		board after = state;
		int role = 1;
		if(node->get_role() == who){
			role = 0;
		}
		// std::cout<<after<<std::endl;
		// std::cout<<"simulation start as: "<<role<<" "<<(role xor 1)<<std::endl;
		while(1){
			//sstd::cout<<role<<" "<<(role xor 1)<<std::endl;
			// std::cout<<after<<std::endl;
			action::place move = random_take_action(after, role);
			if(move == action()) {
				// std::cout<<"simulation end as: "<<role<<std::endl;
				return role;
			}
			else
				move.apply(after);

			move = random_take_action(after, (role xor 1));
			if(move == action()) {
				// std::cout<<"simulation end as: "<<(role xor 1)<<std::endl;
				return (role xor 1);
			}
			else
				move.apply(after);
		}
	}

	virtual std::pair<action, int> selection(const board& state, std::shared_ptr<tree_node> node) {
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
			//std::cout<<after<<"\n\n\n"<<std::endl;
			int score;
			// std::cout<<node->get_role()<<" "<<who<<" "<<move<<std::endl;
			if (move.apply(after) == board::legal){
				if(node->get_role() == who)
					score = node->UCB_score(move);
				else
					score = -node->UCB_score(move);
				if(score > best_score){
					best_move = move;
					best_score = score;
					best_after = after;
				}
			}
			// else{
			// 	std::cout<<move.apply(after)<<std::endl;
			// }
				
		}
		if(best_score == -999999){
			// std::cout<<"can't find legal action as child\n";
			// std::cout<<"play as "<<node->get_role()<<", board: \n";
			// std::cout<<state<<std::endl;
			int win = -1;
			if(node->get_role() == oppo)
				win = 1;
			node->visit_record(win*10);
			return std::pair<action, int>(action(), win);

		}
		if(node->has_child(best_move)){
			// std::cout<<"select\n";
			std::pair<action, int> back_prop;
			back_prop = selection(best_after, node->child(best_move));
			node->visit_record(back_prop.second);
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
			node->visit_record(result);
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
			std::cout<<"can't find opponent's move"<<std::endl;
			std::cout<<"last board:\n";
			std::cout<<last_board<<std::endl;
			std::cout<<"\n\ncurrent board:\n";
			std::cout<<state<<std::endl;
			
			/*new game*/
			// std::cout<<"reset\n";
			MCT.reset_tree(who);
			turn = 0;
		}
		else
			MCT.move_root(oppo_mv);
	}

	virtual action take_action(const board& state) {
		//std::cout<<"take action\n";
		action move;
		//update opponent move if state is not empty board
		if(turn){
			// play as black and not init
			//std::cout<<"update oppo\n";
			board after = state;
			handle_oppo_turn(after);
		}
		else{
			// play as black init
			MCT.reset_tree(who);
			turn = 0;
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
		for (int i=0;i<100;i++) {
			board after = state;
			move = selection(after, MCT.get_root()).first;
			std::cout<<"simulation count: "<<i<<std::endl;
			// std::cout<<"root winrate: "<<MCT.get_root()->get_winrate()<<std::endl;
			// std::cout<<"move UCB score: "<<MCT.get_root()->UCB_score(move)<<std::endl;
		}
		std::cout<<"root winrate: "<<MCT.get_root()->get_winrate()<<std::endl;
		std::cout<<"move UCB score: "<<MCT.get_root()->UCB_score(move)<<std::endl;
		
		board after = state;
		MCT.move_root(move);
		if(move.apply(after) == board::legal){
			last_board = after;
			// std::cout<<"move: "<<move<<std::endl;
			// std::cout<<"root role: "<<MCT.get_root()->get_role()<<std::endl;
			return move;
		}
		else
			return action();
	}

private:
	std::vector<action::place> space;
	std::vector<action::place> oppo_space;
	board::piece_type who;
	board::piece_type oppo;
	tree MCT;
	board last_board;
	int turn;
};


