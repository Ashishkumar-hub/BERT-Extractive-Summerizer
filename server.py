from flask import Flask
from flask import request, jsonify, abort, make_response,render_template
from flask_cors import CORS,cross_origin
import nltk
nltk.download('punkt')
from nltk import tokenize
from typing import List
import argparse
from summarizer import Summarizer, TransformerSummarizer


app = Flask(__name__)
CORS(app)


class Parser(object):

    def __init__(self, raw_text: bytes):
        self.all_data = str(raw_text, 'utf-8').split('\n')

    def __isint(self, v) -> bool:
        try:
            int(v)
            return True
        except:
            return False

    def __should_skip(self, v) -> bool:
        return self.__isint(v) or v == '\n' or '-->' in v

    def __process_sentences(self, v) -> List[str]:
        sentence = tokenize.sent_tokenize(v)
        return sentence

    def save_data(self, save_path, sentences) -> None:
        with open(save_path, 'w') as f:
            for sentence in sentences:
                f.write("%s\n" % sentence)

    def run(self) -> List[str]:
        total: str = ''
        for data in self.all_data:
            if not self.__should_skip(data):
                cleaned = data.replace('&gt;', '').replace('\n', '').strip()
                if cleaned:
                    total += ' ' + cleaned
        sentences = self.__process_sentences(total)
        return sentences

    def convert_to_paragraphs(self) -> str:
        sentences: List[str] = self.run()
        return ' '.join([sentence.strip() for sentence in sentences]).strip()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def convert_raw_text():
    ratio = float(request.args.get('ratio', 0.2))
    min_length = int(request.args.get('min_length', 25))
    max_length = int(request.args.get('max_length', 500))

    data = request.data
    if not data:
        abort(make_response(jsonify(message="Request must have raw text"), 400))

    parsed = Parser(data).convert_to_paragraphs()
    summary = summarizer(parsed, ratio=ratio, min_length=min_length, max_length=max_length)
    summary = summary[25:]
    return jsonify({
        'summary': summary
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='') # bert-large-uncased
    #parser.add_argument('-model', dest='model', default='bert-base-uncased', help='The model to use')
    parser.add_argument('-model', dest='model', default='bert-large-uncased', help='The model to use')
    parser.add_argument('-transformer-type',
                        dest='transformer_type', default=None,
                        help='Huggingface transformer class key')
    parser.add_argument('-transformer-key', dest='transformer_key', default=None,
                        help='The transformer key for huggingface. For example bert-base-uncased for Bert Class')
    parser.add_argument('-greediness', dest='greediness', help='', default=0.45)
    parser.add_argument('-reduce', dest='reduce', help='', default='mean')
    parser.add_argument('-hidden', dest='hidden', help='', default=-2)
    parser.add_argument('-port', dest='port', help='', default=5000)
    parser.add_argument('-host', dest='host', help='', default='0.0.0.0')

    args = parser.parse_args()

    if args.transformer_type is not None:
        print(f"Using Model: {args.transformer_type}")
        assert args.transformer_key is not None, 'Transformer Key cannot be none with the transformer type'

        summarizer = TransformerSummarizer(
            transformer_type=args.transformer_type,
            transformer_model_key=args.transformer_key,
            hidden=int(args.hidden),
            reduce_option=args.reduce
        )

    else:
        print(f"Using Model: {args.model}")

        summarizer = Summarizer(
            model=args.model,
            hidden=int(args.hidden),
            reduce_option=args.reduce
        )

    app.run(host=args.host, port=int(args.port))


''' "The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price. The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal. Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008. Real estate firm Tishman Speyer had owned the other 10%. The buyer is RFR Holding, a New York real estate company. Officials with Tishman and RFR did not immediately respond to a request for comments. It's unclear when the deal will close. The building sold fairly quickly after being publicly placed on the market only two months ago. The sale was handled by CBRE Group. The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building. The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028. Meantime, rents in the building itself are not rising nearly that fast. While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities. Still the building is among the best known in the city, even to people who have never been to New York. It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top. It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day. The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices. Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest. Blackstone Group (BX) bought it for $1.3 billion 2015. The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself. Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete. Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title.",
    
    "ratio": 2.5,
    "min_length": 100,
    "max_length": 1000'''