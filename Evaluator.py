from treys import Card
from treys import Evaluator
from treys import Deck

deck = Deck()

#cutoff values:
#safe is better than half of the 2-pair and below card hand ranks
#the odds of someone getting a better hand is only 27.05 percent in a 7-card hand
safe = 2896
#lowest 2-pair has 38.8% odds of someone getting a better hand
medium = 4284
#lowest 1-pair has 82.6% odds of someone getting a better hand
risky = 6163

#weird calculations were done dw about these
safe_perc = 0.1
medium_perc = 0.2
risky_perc = 0.3



#used to form current deck
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
suits = ['s', 'h', 'd', 'c']  # Spades, hearts, diamonds, clubs


#cards.append(self.draw())

board = []
board_test = ['Ts', '5d', '7s', '9c', 'Qd']
board.extend((Card.new('Ts'), Card.new('5d'), Card.new('7s'), Card.new('9c'), Card.new('Qd')))
Card.print_pretty_cards(board)

hand = []
hand_test = ['Qc', '7d']
hand.extend((Card.new('Qc'), Card.new('7d')))

hand2 = []
hand2.extend((Card.new('Qs'), Card.new('2h')))
hands = [hand,  hand2]

curr_dec = [rank + suit for rank in ranks for suit in suits]
curr_dec = [i for i in curr_dec if i not in board_test]

evaluator = Evaluator()
rank_num = evaluator.evaluate(board, hand)
rank_class = evaluator.get_rank_class(rank_num)

print("my hand")
Card.print_pretty_cards(board+hand)
print(f"Current Hand: {evaluator.class_to_string(rank_class)}")
print(f"Rank (out of 7462): {rank_num}")


print("==============")

curr_dec = [i for i in curr_dec if i not in hand_test]

count = 0
enemy_win = 0

for idx, card1 in enumerate(curr_dec):
    for idx2, card2 in enumerate(curr_dec[idx+1:]):   
        #testing out every other possible combination of hands to see how our hand compares
        hand_temp = [Card.new(card1), Card.new(card2)]
        #Card.print_pretty_cards(hand_temp)
        #total percentage 
        rank_temp = evaluator.evaluate(board, hand_temp)
        if rank_temp <= rank_num:
            Card.print_pretty_cards(hand_temp+board)
            enemy_win += 1
        #print(evaluator.evaluate(board, hand_temp))
        count += 1

print(f"total combination of hands that another player can have: {str(count)}")
print(f"total percentage of hands that could be better than ours: {str(enemy_win/count)}")
print(f"Current Hand: {evaluator.class_to_string(rank_class)}")
print(f"Rank (out of 7462): {rank_num}")
print(curr_dec)

if enemy_win/count > medium_perc:
    print("dont bet! the odds are too low for you")

if enemy_win/count < medium_perc:
    print("Go for it! the odds are looking good for you")













