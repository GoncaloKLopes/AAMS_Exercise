import sys


class Agent:
    def __init__(self, state):
        pass

    def update(self, observation):
        pass

    def decide(self, behavior):
        return '1'

class Lottery:
	def __init__(self, id, options):
		self.id = id
		self.options = options

	def get_utility(self):
		pass

class Option:
	def __init__(self, id, prob, lottery):
		self.id = id
		self.prob = prob
		self.lottery = lottery

	def get_utility(self):
		pass

def option_from_string(string):
	"""
		Returns an Option object from a string 
	"""
	

args = sys.stdin.readline().split(' ')
agent = Agent(args[0])
size = 1 if len(args) <= 2 else int(args[2])
for i in range(0, size):
    if i != 0:
        agent.update(sys.stdin.readline())
    sys.stdout.write(agent.decide(args[0]) + '\n')
