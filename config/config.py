import configparser

config = configparser.ConfigParser()
config.read('config.ini')

mean = config['Train']['Dashboard']['mean']
std = config['Train']['Dashboard']['std']

print(mean)
print(std)