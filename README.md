# Le fonctionnement des LLM de type GPT

# Introduction

Dans ce guide je vais essayer de vous faire comprendre de façon détaillée comment fonctionne le système GPT (**generative pre-trained transformer**) il en existe de plein de type qui ont des fonctions différentes (BioGPT (biomédecine) [ProGPT2](https://huggingface.co/nferruz/ProtGPT2), [ChatGPT](https://chatgpt.com/) (modèle général) mais nous allons nous focaliser plutôt sur une architecture proche de GPT 2.

Le seul prérequis est d’avoir déjà utilisé [ChatGPT](https://chatgpt.com/). L’idée de ce guide est d’être simple et détaillé à la fois en vulgarisant le moins possible.

# Les Datasets

## Qu’est-ce qu’un dataset et à quoi ça sert ?

Un dataset est un **ensemble de données structurées**, comme par exemple :

- un ensemble de photos de pingouins 🐧,

- un ensemble de tous les écrits de Shakespeare 📖,

- ou encore une liste de tous les repas d’un individu pendant un an 🥘.

Ces données peuvent être **labellisées**, c’est-à-dire que chaque donnée est associée à une étiquette. Si on reprend l’exemple du dataset contenant tous les repas d’un individu pendant un an, chaque repas peut être labellisé en associant le jour et le moment de la journée où le repas a été pris.

Les datasets sont utilisés dans divers domaines comme le **machine learning** ou encore la **création de bases de données**.

Quand un dataset contient un ensemble de textes on parle de **corpus**.
## Quel rôle jouent les datasets dans l’entraînement des modèles GPT ?


Les datasets sont le composant le plus important des modèles de langage car ils forgent **le style et la qualité du langage** et **les connaissances d’un modèle**.
## WebText

Les modèles au-delà de GPT 2 de chez open ai n’ont pas été entraînés sur des datasets comme Wikipédia mais sur **WebText**.

WebText contient le contenu de tous les **liens sortants de Reddit avec au moins 3 karmas (likes)**.

WebText est basé sur **45 millions de liens**, dont **8 millions de documents** pour un total de **40 Go de texte**.

OpenAI a fait ça pour mettre l’**accent sur la qualité des données**.

Il est important de noter que tous les documents provenant de Wikipédia ne font pas partie de WebText.

Cela est dû à deux raisons principales :

- Pour diversifier la langue (Wikipédia a un langage neutre et très structuré),

- Pour éviter les répétitions de données.

WebText **n’est pas open source** (public), mais il existe des datasets similaires open source comme [OpenWebText](https://github.com/jcpeterson/openwebtext) ou encore [The Pile](https://pile.eleuther.ai/).

## Liste de datasets que vous pouvez utiliser 

On choisit souvent un dataset en fonction de **sa taille** et du **type de langage** souhaité.

Datasets légers: 

- **Tiny Shakespeare** : [https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (léger, il est conseillé pour faire des petits modèles)

- **Wikipedia** : [https://huggingface.co/datasets/openskyml/wikipedia](https://huggingface.co/datasets/openskyml/wikipedia) (disponible en plusieurs langages)

- **Open Artificial Knowledge** : https://oakdataset.org/ (les données ont été récupérées grâce à des IA comme ChatGPT, Claude et Gemini)

Datasets plus lourds: 

- **OpenWebText** : https://github.com/jcpeterson/openwebtext (version open source de WebText de 38 Go)

- **The Pile** : https://pile.eleuther.ai (corpus de 886 go utilisé pour des modèles comme GPT-Neo et LLaMa)

# Tokenisation 

## La Tokenisation c’est quoi ?

Pour comprendre un texte, notre modèle GPT a besoin d’**effectuer des opérations mathématiques sur les mots**. Mais sur du texte brut c’est juste impossible. Pour cela on a besoin de **tokeniser** notre texte. C’est à dire qu’on va **transformer et découper en unités le texte pour l’adapter** à notre modèle. On dit que le texte est découpé en **token**.

Dans le cas d’un modèle comme GPT, celui-ci va dans un premier temps **prendre un corpus de texte et le diviser en unités**. 
Chaque unité est ajoutée à une **liste de vocabulaire** **avec un identifiant** s’il n’existe pas. 

Puis notre modèle est **prêt** à tokeniser. Le texte sera découpé en unités chaque unité dans le texte est remplacée par le numéro associé. 

![Miro Image](https://miro.medium.com/v2/resize:fit:828/format:webp/1*gWP5Whykah1101EpYy17qQ.png)

*Source Image : https://teetracker.medium.com/llm-fine-tuning-step-tokenizing-caebb280cfc2

La question principale est donc comment allons-nous découper notre texte. Nous allons explorer les différentes approches qui s’offrent à nous.
## Les tokens spéciaux dans la tokenisation

Dans le vocabulaire en plus des unités il y a des tokens spéciaux en voici une liste:

- `[CLS]` / `[SEP]` Marquent le **début/fin** de la séquence
- `[UNK]` Remplace **les unités inconnues** dans le vocabulaire
- `[PAD]` Lors de l’entraînement d’un modèle si l'on veut pouvoir gérer **plusieurs phrases** en même temps il faut qu’elles aient la **même longueur**. Pour cela, on ajoute le token `[PAD]` à la fin des phrases les plus courtes. Il faudra quand même spécifier au modèle de ne pas faire attention à `[PAD]` et l’ignorer dans son entraînement. 

Exemple `[PAD]`:
```
séquence 1 : [[CLS], "Bonjour", "tout", "le", "monde", [SEP]]  
séquence 2 : [[CLS], "Salut", [SEP], [PAD], [PAD], [PAD]]
```

### Bonus technique:

Pourquoi faut il que les phrases aient la même longueur quand elles sont par lot ?

1. Les tenseurs **ne peuvent pas avoir des lignes de longueurs différentes**

Les tenseurs sont des **types d’objets optimisés pour les calculs en python** et ils ne peuvent pas contenir des lignes avec des tailles différentes.

2. On doit **connaître à l’avance les dimensions d’une matrice** si on veut faire un produit matriciel ou un softmax

Nous verrons plus tard que pour notre modèle Transformer le modèle doit connaître à l’avance la **taille du batch** (la taille du lot que notre modèle va ’absorber’ à chaque étape de notre entraînement), justement la **taille des séquences dans les lots** et enfin les **dimensions vectorielles de chaque token** mais nous verrons cela plus en détail plus tard.

### La Tokenisation par mot

Une méthode est de **diviser notre texte par mots**.

On commence par définir notre vocabulaire pour cela, imaginons que nous prenons juste une phrase

```python
"Le chat mange la souris"
```

Le vocabulaire sera donc : 

```python
{0: 'souris', 1: 'chat', 2: 'le', 3: 'la', 4: '.', 5: 'mange', 6: '[UNK]', 7: '[PAD]', 8: '[CLS]', 9: '[SEP]'}
```

Imaginons que nous essayons de tokeniser avec notre modèle la phrase <mark style="background: #BBFABBA6;">"La souris mange le chat."</mark>, le résultat sera donc <mark style="background: #BBFABBA6;">[8, 2, 1, 6, 6, 2, 6, 4, 9]</mark>

```python
➡️ "La souris mange le chat."
➡️ ['[CLS]', 'la', 'souris', 'mange', 'le', 'chat', '.', '[SEP]']
➡️ [8, 3, 0, 5, 2, 1, 4, 9]
```

### Les problèmes de cette méthode

Prenons une autre phrase :

<mark style="background: #FFB86CA6;">"Le chat est sur le canapé."</mark>

Le résultat de la tokenisation sera :  
<mark style="background: #FF5582A6;">['[CLS]', 'le', 'chat', '[UNK]', '[UNK]', 'le', '[UNK]', '.', '[SEP]']</mark>  

ce qui donne :  
<mark style="background: #CACFD9A6">[8, 2, 1, 6, 6, 2, 6, 4, 9]</mark>

Si un mot n’est **pas exactement dans le vocabulaire**, il sera totalement **perdu et remplacé** par `[UNK]`ce qui constitue une **perte d’information** pour notre modèle. Le **vocabulaire a donc besoin de contenir énormément de mots**, ce qui le rendrait énorme, ce qui est un inconvénient pour les calculs car il occuperait **beaucoup de mémoire**. De plus le modèle **ne comprendrait pas les mots avec des fautes d’orthographe**. 

### Bonus : Implémentation en python

```python
import re

#Prenons une seule phrase comme corpus
corpus = "Le chat mange la souris."
  
#On va séparer les ponctuations
corpus = re.sub(r"([?!.,;:])", r" \1 ", corpus)

#Imaginons qu’une unité corresponde à un mot
#Commençons par définir un vocabulaire associant chaque mot à un numéro
vocabulaire = {i: word for i, word in enumerate(set(corpus.lower().split()))}
#⚠️ J’utilise set donc l’ordre des tokens normaux n’est pas déterministe (car un set en Python est implémenté comme une table de hachage)

#Puis ajoutons les tokens spéciaux
vocabulaire[len(vocabulaire)] = '[UNK]' #pour les mots inconnus
vocabulaire[len(vocabulaire)] = '[PAD]' #pour le padding
vocabulaire[len(vocabulaire)] = '[CLS]' #pour le début de phrase
vocabulaire[len(vocabulaire)] = '[SEP]' #pour la fin de phrase

#Notre vocabulaire est donc {0: 'souris', 1: 'chat', 2: 'le', 3: 'la', 4: '.', 5: 'mange', 6: '[UNK]', 7: '[PAD]', 8: '[CLS]', 9: '[SEP]'}

vocabulaire_inverse = {word: i for i, word in vocabulaire.items()}
#{'souris': 0, 'chat': 1, 'le': 2, 'la': 3, '.': 4, 'mange': 5, '[UNK]': 6, '[PAD]': 7, '[CLS]': 8, '[SEP]': 9}

#puis définissons une fonction qui va découper et transformer notre texte
def tokenize(text):
    #On va d’abord le mettre en minuscule
    text = text.lower()

    #On va ensuite séparer les ponctuations
    text = re.sub(r"([?!.,;:])", r" \1 ", text)

    #On va ensuite séparer les mots
    text = text.split()

    #on va ajouter les tokens spéciaux
    text = ['[CLS]'] + ['[UNK]' if w not in vocabulaire_inverse else w for w in text] + ['[SEP]']

    #On va ensuite transformer les mots en id
    text_tokenise = [vocabulaire_inverse[word] for word in text]

    return text_tokenise
  
print(tokenize("La souris mange le chat."))
#['[CLS]', 'la', 'souris', 'mange', 'le', 'chat', '.', '[SEP]']  => [8, 3, 0, 5, 2, 1, 4, 9]

print(tokenize("Le chat est sur le canapé."))
#['[CLS]', 'le', 'chat', '[UNK]', '[UNK]', 'le', '[UNK]', '.', '[SEP]'] => [8, 2, 1, 6, 6, 2, 6, 4, 9]
```

## La Tokenisation par caractère

Une autre approche est la tokenisation par caractère. C’est à dire de **diviser le texte par caractère à la place de mots**. Cette approche **réduirait considérablement la taille du vocabulaire et garantit que la plupart du vocabulaire soit reconnu**.

### Les Inconvénients de cette approche 

Il y a deux gros inconvénients:

- Les séquences deviennent très longues car elles sont découpés plus finement.  
➡️ <mark style="background: #FFB86CA6;">"Le chat est sur le canapé."</mark>
➡️ <mark style="background: #CACFD9A6;">[36, 17, 10, 0, 8, 13, 6, 25, 0, 10, 24, 25, 0, 24, 26, 23, 0, 17, 10, 0, 8, 6, 19, 6, 21, 32, 0, 4, 0, 37]</mark>

- C’est plus compliqué pour un modèle de comprendre la relation entre les caractères que la relation entre les mots.
### Bonus : Implémentation en python

```python
import re

#Prenons comme corpus seulement une phrase
corpus = "Hier, au zoo, j’ai vu dix guépards, cinq zébus, un yak et le wapiti fumer. ?!…"
#Cette phrase a la particularité de contenir toutes les lettres de l'alphabet et quelques ponctuations

#On va séparer les ponctuations
corpus = re.sub(r"([?!.,;:])", r" \1 ", corpus)

#On commence par définir un vocabulaire en associant un identifiant à chaque caractère
vocabulaire = {i : word for i, word in enumerate(sorted(set(corpus.lower())))}
#{0: ' ', 1: '!', 2: "'", 3: ',', 4: '.', 5: '?', 6: 'a', 7: 'b', 8: 'c', 9: 'd', 10: 'e', 11: 'f', 12: 'g', 13: 'h', 14: 'i', 15: 'j', 16: 'k', 17: 'l', 18: 'm', 19: 'n', 20: 'o', 21: 'p', 22: 'q', 23: 'r', 24: 's', 25: 't', 26: 'u', 27: 'v', 28: 'w', 29: 'x', 30: 'y', 31: 'z', 32: 'é', 33: '…'}

#Puis ajoutons les tokens spéciaux
vocabulaire[len(vocabulaire)] = '[UNK]' #pour les mots inconnus
vocabulaire[len(vocabulaire)] = '[PAD]' #pour le padding
vocabulaire[len(vocabulaire)] = '[CLS]' #pour le début de phrase
vocabulaire[len(vocabulaire)] = '[SEP]' #pour la fin de phrase

#Nous nous retrouvons donc avec un vocabulaire de 37 éléments
#{0: ' ', 1: '!', 2: "'", 3: ',', 4: '.', 5: '?', 6: 'a', 7: 'b', 8: 'c', 9: 'd', 10: 'e', 11: 'f', 12: 'g', 13: 'h', 14: 'i', 15: 'j', 16: 'k', 17: 'l', 18: 'm', 19: 'n', 20: 'o', 21: 'p', 22: 'q', 23: 'r', 24: 's', 25: 't', 26: 'u', 27: 'v', 28: 'w', 29: 'x', 30: 'y', 31: 'z', 32: 'é', 33: '…', 34: '[UNK]', 35: '[PAD]', 36: '[CLS]', 37: '[SEP]'}

vocabulaire_inverse = {word: i for i, word in vocabulaire.items()}

#puis définissons une fonction qui va découper et transformer notre texte
def tokenize(text):
    #On va d’abord le mettre en minuscule
    text = text.lower()

    #On va ensuite séparer les ponctuations
    text = re.sub(r"([?!.,;:])", r" \1 ", text)

    #on va ajouter les tokens spéciaux
    text = ['[CLS]'] + ['[UNK]' if w not in vocabulaire_inverse else w for w in text] + ['[SEP]']

    #On va ensuite transformer les mots en id
    text_tokenise = [vocabulaire_inverse[word] for word in text]

    return text_tokenise

print(tokenize("Le chat mange la souris."))
# => ['[CLS]', 'l', 'e', ' ', 'c', 'h', 'a', 't', ' ', 'm', 'a', 'n', 'g', 'e', ' ', 'l', 'a', ' ', 's', 'o', 'u', 'r', 'i', 's', ' ', '.', ' ', '[SEP]']
# => [36, 17, 10, 0, 8, 13, 6, 25, 0, 18, 6, 19, 12, 10, 0, 17, 6, 0, 24, 20, 26, 23, 14, 24, 0, 4, 0, 37]

print(tokenize("Le chat est sur le canapé."))
# => ['[CLS]', 'l', 'e', ' ', 'c', 'h', 'a', 't', ' ', 'e', 's', 't', ' ', 's', 'u', 'r', ' ', 'l', 'e', ' ', 'c', 'a', 'n', 'a', 'p', 'é', ' ', '.', ' ', '[SEP]']
# => [36, 17, 10, 0, 8, 13, 6, 25, 0, 10, 24, 25, 0, 24, 26, 23, 0, 17, 10, 0, 8, 6, 19, 6, 21, 32, 0, 4, 0, 37]
```

## Tokenisation BPE (Byte Pair Encoding) 

Le BPE est une méthode de tokenisation par **subwords** (c’est-à-dire **des sous-mots**). Au début, **BPE découpe en caractère et petit à petit ajoute les paires les plus fréquentes**. Ainsi, BPE a pour objectif de réduire le nombre de tokens nécessaire pour tokeniser une phrase sans perdre en précision.

On commence avec un vocabulaire avec tous les caractères de base du corpus et on traite le corpus pour qu’il soit représenté comme une liste de tokens

```python
Corpus = "hug", "pug", "pun", "bun", "hugs"

New_Corpus = ['h', 'u’, ’g’, ’</w>’,’p’, ’u’, ’g’, ’</w>’,’p’, ’u’, ’n’, ’</w>’,’b’, ’u’, ’n’, ’</w>’,’h’, ’u’, ’g’, ’s’, ’</w>’]

notre_vocabulaire_de_base = ["b", "g", "h", "n", "p", "s", "u"]
```

Puis on se met à compter les paires et leurs fréquences

| Paire       | Fréquence |
| ----------- | --------- |
| (u, g)      | 3         |
| (h, u)      | 2         |
| (g, `</w>`) | 2         |
| (p, u)      | 2         |
| (u, n)      | 2         |
| (n, `</w>`) | 2         |
| (b, u)      | 1         |
| (g, s)      | 1         |
| (s, `</w>`) | 1         |

On observe que la paire la plus fréquente est (u, g) donc on la fusionne dans notre corpus.

```python
New_Corpus = ['h', 'ug', '</w>','p', 'ug', '</w>','p', 'u', 'n', '</w>','b', 'u', 'n', '</w>','h', 'ug', 's', '</w>']
```

Puis on recommence 

| Paire        | Fréquence |
| ------------ | --------- |
| (h, ug)      | 2         |
| (ug, `</w>`) | 2         |
| (u, n)       | 2         |
| (n, `</w>`)  | 2         |
| (p, ug)      | 1         |
| (s, `</w>`)  | 1         |
| (ug, s)      | 1         |
| (p, u)       | 1         |
| (b, u)       | 1         |

On voit que plusieurs paires ont la même fréquence donc on décide de prendre la première dans l’ordre alphabétique c’est à dire (h, ug), puis fusionner dans notre corpus.

```python
New_Corpus = ['hug', '</w>', 'p', 'ug', '</w>','p', 'u', 'n', '</w>','b', 'u', 'n', '</w>','hug', 's', '</w>']
```

Puis on répète le processus jusqu’à atteindre la taille de vocabulaire souhaitée. Dans cet exemple, une taille de vocabulaire de 14 tokens.

```python
New_Corpus = ['hug', '</w>', 'p', 'ug', '</w>','p', 'un</w>','b','hug', 's', '</w>']

vocab = {0: "</w>", 1: "b", 2: "g", 3: "h", 4: "n", 5: "p", 6: "s", 7: "u", 8: 'ug', 9: 'hug', 10: 'un</w>', 11: "[CLS]", 12: "[SEP]", 13: "[UNK]", 14: "[PAD]"}
```

Maintenant imaginons que je veuille tokeniser le mots "hugs"

➡️ `["[CLS]", "hug", "s", "</w>", "[SEP]"]`
➡️  <mark style="background: #CACFD9A6;">[11, 9, 6, 0, 12]</mark>

Le résultat est donc `[11, 9, 6, 0, 12]`

La première fois que j’ai entendu parler du terme BPE, c’était dans une vidéo YouTube qui parlait de la compression de données car à la base BPE était une technique de compression.

## Enfin la tokenisation Byte-Level BPE

La tokenisation Byte-Level BPE est une variante de la tokenisation Byte Pair Encoding. Sauf que contrairement au BPE, l'algorithme n’est pas appliqué sur du **texte brut** mais sur des octets. **Chaque caractère** (que ce soit les émojis ou des lettres accentuées) est **converti en byte**, une valeur numérique entre 0 et 255.  

C’est la tokenisation Byte-Level BPE qui est **utilisée dans les modèles GPT**. Cela signifie que GPT n’apprend pas à partir de mots ou de lettres mais à partir de séquences de bytes.

**Pourquoi ce choix ?**

Tous les textes peuvent être représentés sous forme de bytes. Donc pas de OOV, c’est à dire de *Out of Vocabulary*, **aucun mots qui ne sera pas pris en compte par le modèle**.

**Il n’existe pas de symbole ou caractère que l’on ne puisse pas représenter en bytes**. Comme notre vocabulaire contient au minimum les 256 bytes possibles, il n’y aura jamais de *Out of Vocabulary*.

En contrepartie, il y a **plus de tokens par phrase** et il est **plus difficile pour le modèle de comprendre le sens des mots** car les mots sont sous forme de séquences de bytes.

## Fonctionnement 

1. On commence avec un vocabulaire contenant les 256 bytes possibles + les tokens spéciaux. 

GPT-2 utilise un seul vrai jeton spécial "\`\`"  qui marque la fin d'un texte. Les espaces sont représentés par "Ġ+mots" (ex: "Ġvoiture") et les sauts de lignes par "Ċ".

2. Le corpus est encodé en bytes (UTF-8).

3. Ces bytes sont regroupés par paires grâce à l’algorithme BPE (Byte Pair Encoding).

4. Le processus recommence jusqu’à un nombre de token précis dans le vocabulaire

### Bonus : Implémentation en python

On commence par installer tiktoken une librairie développée et utilisée par open ai pour faire de la tokenisation de type Byte Level BPE.

```
!pip install tiktoken
```

Tiktoken est très simple à utiliser tout en étant très performant.

```python
import tiktoken

#on utilise le tokenizer de gpt2
enc = tiktoken.get_encoding("gpt2")

#phrase que nous allons tokeniser
sentence = "Le chat est sur le canapé 🙀."

#on va tokeniser notre phrase
text_tokenize = enc.encode(sentence)
print(text_tokenize)
#[3123, 8537, 1556, 969, 443, 460, 499, 2634, 12520, 247, 222, 13]

#on peut la détokeniser
decoded = enc.decode(text_tokenize)
print(decoded)
#Le chat est sur le canapé 🙀.

#on peut avoir un aperçu de la tokenisation
for t in text_tokenize:
    print(f"{t} → '{enc.decode([t])}'", end=", ")
#3123 → 'Le', 8537 → ' chat', 1556 → ' est', 969 → ' sur', 443 → ' le', 460 → ' can', 499 → 'ap', 2634 → 'é', 12520 → ' �', 247 → '�', 222 → '�', 13 → '.',

#on observe que l’emoji 🙀 est représenté par plusieurs bytes car il n’est pas entièrement dans le vocabulaire donc pour le tokeniser il est divisé en 3 sequences de bytes.
```

# Embeddings : Représentation vectorielle des tokens

On passe maintenant à une des parties les plus intéressantes le **word embedding**.

Le word embedding permet de **transformer des mots en vecteur pour capturer leur sens**. Cela permet par exemple de faire des opérations avec les mots comme: 

```python
roi - homme + femme ≈ reine
 ```

🫷  **L'embedding ne s'applique pas uniquement aux mots mais peut aussi s'appliquer sur des tokens**

Mais comment capturer le sens d'un mot ?

L'embedding est basé sur l'**hypothèse distributionnelle**, une théorie énoncée pour la première fois par Zellig Harris dans [<u>Distributional Structure</u>](https://www.tandfonline.com/doi/pdf/10.1080/00437956.1954.11659520) en 1954 qui dit que *"**les différences de sens sont corrélées aux différences de distribution**"*. En d'autres termes **les mots qui apparaissent dans le même contexte ont tendance à avoir des significations similaires** donc à partir du moment où l'on a le contexte d'un mots on peut avoir son sens. Par exemple si le mot "**voiture**" est statistiquement **entouré des mêmes mots** que "**automobile**" ils ont un **sens proche**.

## La cooccurrence 

La **cooccurrence** désigne le fait que **deux mots apparaissent dans un même contexte**. 

Dans notre exemple précédent on dit que "**voiture**" est statistiquement **entouré des même mots** que "**automobile**" donc "automobile" et "voiture" **sont en cooccurrence**. 

À partir de l'**hypothèse distributionnelle** on peut déjà faire de l'embedding simple. Prenons 2 phrases.

```python
"Le chat est sur le canapé.", "Le chat mange la souris."
```

Maintenant nous allons pour chaque mot noter son contexte textuel avec une fenêtre de taille 1 c'est à dire que le contexte est de 1 mot autour du mot dont on cherche le contexte. Dans cet exemple imaginons que on cherche le contexte de mot "**est**". Nous voyons alors que les mots "**chat**" et "**sur**" font parti du contexte du mot "**est**". 

```python
Le "chat" *est* "sur" le canapé.
```

Voici un tableau qui montre le contexte de chaque mot dans nos deux phrases, on peut dire que c'est une matrice de cooccurrence. Pour chaque mot de chaque colonne, à chaque fois qu'un mot est observé dans son contexte on ajoute 1 à la ligne correspondante. 

|        | le  | chat | est | mange | sur | la  | canapé | souris |
| ------ | --- | ---- | --- | ----- | --- | --- | ------ | ------ |
| le     | 0   | 2    | 0   | 0     | 1   | 0   | 1      | 0      |
| chat   | 2   | 0    | 1   | 1     | 0   | 0   | 0      | 0      |
| est    | 0   | 1    | 0   | 0     | 1   | 0   | 0      | 0      |
| mange  | 0   | 1    | 0   | 0     | 0   | 1   | 0      | 0      |
| sur    | 1   | 0    | 1   | 0     | 0   | 0   | 0      | 0      |
| la     | 0   | 0    | 0   | 1     | 0   | 0   | 0      | 1      |
| canapé | 1   | 0    | 0   | 0     | 0   | 0   | 0      | 0      |
| souris | 0   | 0    | 0   | 0     | 0   | 1   | 0      | 0      |
À partir de cette matrice, chaque mot peut être représenté par un vecteur en 8 dimensions, correspondant aux 8 mots du vocabulaire.

Par exemple pour le mot **"est"** son vecteur est : `[0, 1, 0, 0, 1, 0, 0, 0]`.

Le **principal problème** de cette approche est qu'elle **favorise les mots les plus fréquents**. **Les mots qui apparaissent le plus souvent** comme 'le' seront en **cooccurrence avec la plupart des mots.** Pourtant ils n'apportent pas beaucoup de sens à la phrase. Cette approche **ne permet pas de mettre en valeur les mots avec le plus d'information sémantique.**

## Stop word

Les **mots** qui apportent **très peu d'informations sémantiques** sont appelés des **mots vides**. Pour **certains types d'embeddings**, ils peuvent **tout simplement être supprimés**, mais ce n'est pas le cas pour les modèles de langage, car on veut qu'ils soient capables d'interpréter le plus de mots possible.

### TF-IDF (Term Frequency - Inverse Document Frequency) 

Les matrices de cooccurrence brute **survalorisent** donc les **mots très fréquents** comme "le" ou "et" au **détriment des mots** qui apporteraient **plus d'information** sémantique à la phrase.

Pour **palier ce problème** apparaît le **concept de TF-IDF** (Term Frequency - Inverse Document Frequency). 

Pour faire simple avec le concept de TF-IDF **plus un mot est fréquent dans la phrase et plus il est rare dans l’ensemble du corpus, plus il est jugé important**.

Si on reprends nos deux phrases.

```python
"Le chat est sur le canapé.", "Le chat mange la souris."
```

Si on applique TF-IDF on arrive à ce résultat

```python
canapé   -> [0.42519636 0.        ]
chat     -> [0.30253071 0.35520009]
est      -> [0.42519636 0.        ]
la       -> [0.         0.49922133]
le       -> [0.60506143 0.35520009]
mange    -> [0.         0.49922133]
souris   -> [0.         0.49922133]
sur      -> [0.42519636 0.        ]

#Cet exemple est exagéré, car l'algorithme TF-IDF doit être appliqué sur un corpus plus grand que deux phrases.
```

L'approche de la cooccurrence associée au concept de TF-IDF est une approche simple de l'embedding. Il y a une augmentation rapide du nombre de dimensions et les vecteurs sont creux, c'est-à-dire que la plupart des informations que contiennent les vecteurs sont des 0.

Bonus : Comment calculer le TF-IDF 

Le **TF-IDF** est le produit de **TF × IDF**.

$\mathrm{TFIDF}(t, d, D) = \mathrm{TF}(t, d) \times \mathrm{IDF}(t, D)$

Dans cette expression:

- **$t$** représente un mots 
- **$d$** représente le document
- **$D$** représente l'ensemble du corpus

La **term frequency** est calculé grâce à cette expression:

$\mathrm{TF}(t, d) = \frac{\text{Nombre d'occurrences du terme } t \text{ dans le document } d}{\text{Nombre total de mots dans le document } d}$

L'**inverse document frequency** est calculé grâce à cette expression:

$\text{IDF}(t, D) = \log\left(\frac{N}{\text{df}(t)}\right)$

Dans cette expression:

- **$N$** le nombre de documents dans le corpus
- **$df(t)$** représente le nombre de documents dans lequel le mot $t$ fait son apparition

## Les embeddings appris   

Avec les **embeddings appris**, chaque **mot** se voit **attribuer** un vecteur **fixe** **lors de son apprentissage**.

Il existe **deux méthodes d'apprentissage populaires**. 

**CBOW (Continuous Bag of Words)**: CBOW cherche à prédire un mot cible à partir de son contexte textuel.

**Skip-gram**: C'est l'inverse de CBOW, avec Skip-gram on cherche à prédire le contexte textuel d'un mot cible.

Un des modèles d'embedding appris les plus populaires est le modèle **Word2Vec** développé par Google.

### Les Inconvénients de cette approche

On observe **deux inconvénients** principaux à cette approche.

1. Cette approche ne gère pas le **polysémantisme**, le mot souris sera le même qu'il désigne l'animal ou le périphérique de l'ordinateur. 

2. Avec cette approche la **taille du vocabulaire** est verrouillée, pas de *Out of Vocabulary*. 

Nous ne rentrerons **pas plus dans les détails** car les embeddings appris ne sont pas les plus adaptés **pour les modèles de langage**. Néanmoins pour en savoir plus sur le fonctionnement je vous conseille cet [article](https://datascientest.com/nlp-word-embedding-word2vec).

🚧En travaux 🚧
