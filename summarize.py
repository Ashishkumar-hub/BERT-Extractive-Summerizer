from summarizer import Summarizer
import argparse


def run():
    parser = argparse.ArgumentParser(description='Process and summarize lectures')
    parser.add_argument('-path', dest='path', default='input.txt', help='File path of lecture')
    parser.add_argument('-model', dest='model', default='bert-large-uncased', help='')
    parser.add_argument('-hidden', dest='hidden', default=-2, help='Which hidden layer to use from Bert')
    parser.add_argument('-reduce-option', dest='reduce_option', default='mean', help='How to reduce the hidden layer from bert')
    parser.add_argument('-greedyness', dest='greedyness', help='Greedyness of the NeuralCoref model', default=0.45)
    args = parser.parse_args()

    if not args.path:
        raise RuntimeError("Must supply text path.")

    with open(args.path) as d:
        text_data = d.read()

    model = Summarizer(
        model=args.model,
        hidden=args.hidden,
        reduce_option=args.reduce_option
    )

    print(model(text_data))


if __name__ == '__main__':
    run()

'''The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction 
of its previous sales price. Mubadala, an Abu Dhabi investment fund, purchased 90% of the building
for $800 million in 2008. The building sold fairly quickly after being publicly placed on the market
only two months ago. The incentive to sell the building at such a huge loss was due to the soaring 
rent the owners pay to Cooper Union, a New York college, for the land under the building. 
The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
'''