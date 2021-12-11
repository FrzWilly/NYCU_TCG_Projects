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
// UCB exploration ratio
#define C 1.44
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
		space(board::size_x * board::size_y), who(board::empty),
		oppo_space(board::size_x * board::size_y), oppo(board::empty) {
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
	}

	class tree_node{
	public:
		tree_node(board::piece_type role, action::place move) :
			role(role), move(move){
				winrate = INIT_WINRATE;
				visit_count = 0;
			}
		// constructor used for initializing root
		tree_node(board::piece_type role) :
			role(role), move(move){
				winrate = INIT_WINRATE;
				visit_count = 0;
		}

		bool has_child(action::place move){
			return children.count(move);
		}

		tree_node& child(action::place move){
			return children[move];
		}

		void new_child(board::piece_type role, action::place move){
			tree_node new_node(role, move);
			children[move] = new_node;
		}

		void visit_record(){
			visit_count++;
		}

		board::piece_type& get_role(){
			return role;
		}

		int UCB_score(action::place move){
			int c_winrate, c_vcount;
			if(has_child(move)){
				tree_node* chld = &children[move];
				c_winrate = chld->winrate;
				c_vcount = chld->visit_count;
			}
			else{
				c_winrate = INIT_WINRATE;
				c_vcount = 1;
			}
			return c_winrate + C * sqrt(log(visit_count)/c_vcount);
		}
		
	private:
		//int depth;
		board::piece_type role;
		action::place move;
		double winrate;
		int visit_count;
		std::map<action::place, MCTS_player::tree_node> children;
		tree_node* prev;
	};

	class tree{
	public:
		tree(MCTS_player::tree_node root):
		root(root){};
	private:
		MCTS_player::tree_node root;
	};

	virtual action random_take_action(const board& state) {
		std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

	virtual int simulation(const board& state, tree_node& node) {
		board after = state;
		while(1){
			action::place move = random_take_action(after);
			if(move == action()) break;
		}
	}

	virtual action selection(const board& state, tree_node& node) {
		std::shuffle(space.begin(), space.end(), engine);
		action::place* best_move;
		int best_score = -999999;
		board best_after;
		for (const action::place& move : space) {
			board after = state;
			int score;
			if (move.apply(after) == board::legal){
				if(node.get_role() == who)
					score = node.UCB_score(move);
				else
					score = -node.UCB_score(move);
				if(score > best_score){
					*best_move = move;
					best_score = node.UCB_score(move);
					best_after = after;
				}
			}
				
		}
		if(node.has_child(*best_move)){
			selection(best_after, node.child(*best_move));
		}
		else{
			node.new_child(oppo, *best_move);

			node.visit_record();
		}
		return action();
	}

	virtual action take_action(const board& state) {
		// std::shuffle(space.begin(), space.end(), engine);
		for (const action::place& move : space) {
			board after = state;
			if (move.apply(after) == board::legal)
				return move;
		}
		return action();
	}

private:
	std::vector<action::place> space;
	std::vector<action::place> oppo_space;
	board::piece_type who;
	board::piece_type oppo;
	tree MCT;
};


