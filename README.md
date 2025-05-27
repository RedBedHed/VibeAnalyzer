# VibeChecker
##### *A simple machine learning model for analyzing the emotionality of text with ~87% accuracy.*

#### The Haunting of Hill House
```
Post: "Within, walls continued upright, bricks met neatly, floors were firm, and doors were sensibly shut;
silence lay steadily against the wood and stone of Hill House, and whatever walked there, walked alone"

Vibe: (ó﹏ò｡) | fear
```

#### Lord of The Rings
```
Post: "Then Frodo kissed Merry and Pippin, and last of all Sam, and went aboard; and the sails were drawn up,
and the wind blew, and slowly the ship slipped away down the long grey firth; and the light of the glass of
Galadriel that Frodo bore glimmered and was lost. And the ship went out into the High Sea and passed on into
the West, until at last on a night of rain Frodo smelled a sweet fragrance on the air and heard the sound of
singing that came over the water. And then it seemed to him that as in his dream in the house of Bombadil, the
grey rain-curtain turned all to silver glass and was rolled back, and he beheld white shores and beyond them a
far green country under a swift sunrise. But to Sam the evening deepened to darkness as he stood at the Haven;
and as he looked at the grey sea he saw only a shadow on the waters that was soon lost in the West. There still
he stood far into the night, hearing only the sigh and murmur of the waves on the shores of Middle-earth, and
the sound of them sank deep into his heart. Beside him stood Merry and Pippin, and they were silent. At last
the three companions turned away, and never again looking back they rode slowly homewards; and they spoke no
word to one another until they came back to the Shire, but each had great comfort in his friends on the long
grey road. At last they rode over the downs and took the East Road, and then Merry and Pippin rode on to
Buckland; and already they were singing again as they went. But Sam turned to Bywater, and so came back up the
Hill, as day was ending once more. And he went on, and there was yellow light, and fire within; and the evening
meal was ready, and he was expected. And Rose drew him in, and set him in his chair, and put little Elanor upon
his lap. He drew a deep breath. ‘Well, I’m back,’ he said."

Vibe: (=♡.♡=) | love
```

### Sunrise on the Reaping
```
Post: "Like all the Covey, music in her blood. But not like them, too. Less interested in pretty melodies,
more in dangerous words. The kind that lead to rebel acts. The kind that got her arrested twice."

Vibe: (｡•̀ ᴖ •́ ) | anger
```

# Mockingjay
```
Post: "But one day I’ll have to explain about my nightmares. Why they came. Why they won’t ever really go away.
I’ll tell them how I survive it. I’ll tell them that on bad mornings, it feels impossible to take pleasure in
anything because I’m afraid it could be taken away. That’s when I make a list in my head of every act of
goodness I’ve seen someone do. It’s like a game. Repetitive. Even a little tedious after more than twenty years.
But there are much worse games to play."

Vibe: (,>﹏<,) | sadness
```

## Usage
```
usage: vibe.py [-h] [-t T] [-s S]

options:
  -h, --help  show this help message and exit
  -t T        Test Mode: Performs k-fold cross validation and assesses model accuracy. Input any value.
  -s S        String: A string to analyze for emotionality. Input a string.
```

## Credit

### Vibe.py Implementation
>Author: Ellie Moore

### Training Data Used
>https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset
>Author: Bhavik Jikadara
>License: https://creativecommons.org/licenses/by/4.0/
>Changes Made: None

### References
>https://web.stanford.edu/~jurafsky/slp3/4.pdf
>Authors: Daniel Jurafsky & James H. Martin
