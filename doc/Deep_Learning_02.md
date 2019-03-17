# 自然言語処理
* 日本語や英語など普段使っている言葉を自然言語(Natural Language)と言い、自然言語処理(NLP：Natural Language Processing)は自然言語を処理する分野、つまりコンピュータに人間の言葉を理解させるための技術のことである。
* 自然言語処理が扱う分野は多岐にわたるが、その本質はコンピュータに人間の言葉を理解させることにある。
* コンピュータが理解できる言語と言えば、プログラミング言語やマークアップ言語がある。
* それらの言語はコードの意味が一意に解釈できるように文法が定義されており、コンピュータはそのルールに従ってコードを解析する。
* 一般的なプログラミング言語は機械的で無機質であり、"固い言語"である。
* 一方、自然言語は同じ意味の文章でも様々な表現が可能であり、曖昧さがあったり、柔軟に意味や形が変わることから"柔らかい言語"であると言える。
* このような柔らかい言語である自然言語をコンピュータに理解させるのは難しい。
* 一方、その問題をクリアできれば、人にとって役立つことをコンピュータに行わせることができると言える。
* 実際にそのような例は多くあり、検索エンジンや機械翻訳、質問応答システム、文章の自動翻訳や感情分析などが挙げられる。
* 自然言語は文字によって構成され、また自然言語の言葉の意味は単語によって構成される。
* 単語は意味の最小単位であり、自然言語をコンピュータに理解させるためには、単語の意味を理解させることが重要であると言える。
* コンピュータに単語の意味を理解させるために、単語の意味をうまく捉えた以下の表現方法について考える。
    * シソーラスによる手法
    * カウントベースによる手法
    * 推論ベースの手法(word2vec)


# シソーラス
* 単語の意味を表す方法として、辞書のように1つ1つの単語に対してその意味を説明する方法が考えられる。
* このように単語の意味を定義することでコンピュータで単語の意味を理解する。
* ただし、これまでの自然言語処理の歴史では、人が使う辞書とは異なるシソーラス(thesaurus)と呼ばれるタイプの辞書が使われてきた。
* シソーラスは基本的には類語辞書であり、同じ意味の単語(同義語)、意味の似た単語(類義語)が同じグループに分類されている。
* 例えば、シソーラスを使うことでcarの同義語にはautomobileやmotocarなどが存在することがわかる。
* また、シソーラスでは、「上位と下位」、「全体と部分」などのより細かい関係性が単語の間で定義されている場合がある。
* 例えば、グラフ構造によって、各単語の関係性が定義されているケースがある。
* carという単語の上位概念としてmotor vehicleがあり、下位概念としてSUVやcompactがある。
* このように全ての単語に対して類義語の集合を作り、それぞれの単語の関係をグラフで表現することで、単語間の繋がりを定義することができる。
* この構造を使用することでコンピュータに単語間の関連性を教えることができる。


# 代表的なシソーラス - WordNet
* 自然言語処理分野におて、有名なシソーラスとして、WirdNetがある。
* WordNetはプリンストン大学で1985年に開発がスタートしており、これまで多くの研究に利用されてきた。
* WordNetを使うことで類義語を取得できたり、単語ネットワークを利用したりすることができる。
* さらに単語ネットワークを利用することで単語間の類似度を算出することも可能。


# シソーラスの問題点
* シソーラスにより、多くの単語に対して、同義語や階層構造の関係性を定義することによってコンピュータに単語の意味を教えることができる。
* 一方、それらを人の手でラベル付けすることには以下のような大きな欠点が存在する。
    * 時代変化に伴う対応が困難
        * 自然言語は生きており、絶えず新しい言葉が生まれ、古い言葉は忘れ去られる。
        * また時代変化に伴い、言葉の意味が変化する場合もある。
        * そのような単語の変化に対応するためにはシソーラスを人手で絶えず更新する必要がある。
    * 人手によるラベル付けのコストが高い
        * シソーラスを作るためには多くのコストがかかる。
        * 現存する英単語の総数は1000万語を超えると言われているが、それら全てに単語の関連づけをする必要がある。
    * 単語の細かいニュアンスを表現できない
        * シソーラスでは類義語として似た単語をグループ化するが、実際には似た単語であっても、それぞれニュアンスが少し異なる。
        * 単語ごとの微妙なニュアンスの差異はシソーラスでは表すことができない。
* 上記の通り、シソーラスを使う手法(単語の意味を人手によって定義する方法)には大きな問題がある。
* この問題を避けるため「カウントベースの手法」、ニューラルネットワークを使った「推論ベースの手法」がよく用いられる。
* これらの手法では、大量のテキストデータから自動的に単語の意味を抽出する。


# カウントベースの手法
* カウントベースの手法ではコーパス(corpus)を使用する。
* コーパスとは大量のテキストデータであるが、やみくもに集められたテキストデータではなく、自然言語処理の研究やアプリケーション開発のために目的をもって収集されたテキストデータのことを呼ぶ.
* コーパスはテキストデータに過ぎず、含まれる文章は人の手によって書かれたものである。
* これはコーパスに自然言語に対する人間の知識(文章の書き方、単語の選び方、単語の意味)が含まれていることを意味する。
* カウントベースの手法では人間の知識が詰まったコーパスから自動的かつ効率的にそのエッセンスを抽出することを行う。
* 自然言語処理分野で用いられるコーパスにはテキストデータに対してさらに追加の情報が付与されていることがある。
* 例えば、テキストデータの個々の単語に対して、品詞がラベル付けされているケースなどがある。
* その場合、コンピュータが扱いやすいようにコーパスが機構造などのデータ形式として構造化されていることが一般的である。
* 自然言語処理分野で用いられるコーパスには様々なものが存在し、WikipediaやGoogle Newsなどのテキストデータや、シェクスピアや夏目漱石などの偉大な作家の作品群がコーパスとして有名である。
* ここではまずはじめに1文からなる単純な以下のようなテキストをコーパスとして利用する。
    ```python
    >>> text = 'You say goodbye and I say hello.'
    ```
* 次にPythonの対話モードで小さなテキストデータに前処理(テキストデータを単語に分割し、その分割した単語を単語IDのリストへ変換)を行う。
    ```python
    >>> text = text.lower()
    >>> text = text.replace('.', '. ')
    >>> text
    'you say goodbye and i say hello .'

    >>> words = text,split(' ')
    >>> words
    ['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']
    ```
* 上記では、まずlower()で全て小文字に変換し、ピリオドの前にスペースを挿入する。
    * 実際はより賢く汎用的なやり方として正規表現を使用する方法がある。
    * reモジュールをimportし、re.split('(\W+)?', text)とすることで単語単位に分割できる。
* 次にsplit()でスペースを区切り文字として、単語に分割し、wordsというリストに各単語を要素として保持する。
* これで元の文章を単語のリストとして使用できるようになる。
* ただ単語をテキストのまま操作するのは不便なため、単語にIDを振り、IDのリストとして使用できるようにする。
* まず以下の通り、単語のIDと単語の対応表をPythonのディクショナリで作成する。
    ```python
    >>> word_to_id = {}
    >>> id_to_word = {}
    >>>
    >>> for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word
    ```
* id_to_wordにはキーを単語ID、値を単語として保持し、word_to_idにはキーを単語、値を単語IDとして保持する。
* if文では単語がword_to_idにまだ保持されていない場合にword_to_idの要素数をnew_idとして払い出し、word_to_idにそのnew_idを値として追加する。
* id_to_wordにはそのnew_idをキーとしてwordを追加する。
* 上述のtextの文はid_to_word、word_to_idにそれぞれ以下のように保持される。
    ```python
    >>> id_to_word
    {0:'you', 1:'say', 2:'goodbye', 3:'and', 4:'i', 5:'hello', 6:'.'}
    >>> word_to_id
    {'you':0, 'say':1, 'goodbye':2, 'and':3, 'i':4, 'hello':5, '.':6}
    ```
* これらのディクショナリを使うことにより、単語から単語IDを検索したり、単語IDから単語を検索したりできる。
    ```python
    >>> id_to_word[1]
    'say'
    >>> word_to_id['hello ']
    5
    ```
* 最後に以下の通り、単語のリストwordsを単語IDのリストcorpusに変換し、その結果をNumpy配列にして保持する。
    ```python
    >>> import numpy as np
    >>> corpus = [word_to_id[w] for w in words]
    >>> corpus = np.array(corpus)
    >>> corpus
    array([0,1, 2, 3, 4, 1, 5, 6])
    ```
* for文の内包表記ではwordsの先頭から要素の値を順に取り出してwに格納し、そのwをキーとしてword_to_idの値、つまり単語IDを取得する。
* 取得した単語idはcorpusにリストとして保持される。
* 次の処理でcorpusはnp.array()でNumpy配列に変換される。
* ここまでの処理を前処理として、preprocess()という関数にしてまとめて実装すると、以下のようになる。
    ```python
    def preprocess(text):
        text = text.lower()
        text = text.replace('.', '. ')
        words = text,split(' ')

        word_to_id = {}
        id_to_word = {}
        for word in words:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word
    ```
* この関数を使用すると、コーパスの前処理は以下のようになる。
    ```python
    >>> text = 'You say goodbye and I say hello.'
    >>> corpus, word_to_id, id_to_word = preprocess(text)
    ```
* ここまででコーパスを扱う準備が整ったので、これを用い、単語の意味を抽出する。
* ここでカウントベースの手法を用い、単語をベクトルで表すことを行なう。
* ベクトル表現の例として「色」を考える。
* 色には様々な固有の名前が付けられている一方、RGBの3成分がその色にどれだけ含まれるかによって色を表現することもできる。
* 前者は色の数だけ異なる名前を命名することになるが、後者は色を3次元ベクトルで表すことになる。
* ここで、RGBによる色のベクトル表現の方が正確に色を指定できると言える。
* また例えば、(R, G, B) = (201, 23, 30)であれば、その色の名前を知るよりも、赤系の色だとイメージがしやすいと言える。
* 加えて、色同士の関連性・類似性もベクトル表現の方が容易に判断でき、また定量化も容易にできる。
* このような色のベクトル表現を単語でも行えないかを考える。
* つまり、コンパクトかつ理にかなったベクトル表現を「単語」というフィールドにおいて構築することを考える。
* これが構築できれば、「単語の意味」を的確に捉えたベクトル表現が可能となる。
* このベクトル表現を単語の分散表現と呼ぶ。
* 単語の分散表現では単語を固定長のベクトルで表現し、そのベクトルは密なベクトル(ベクトルの各要素が0でない実数で構成されるベクトル)で表現される。
* これまでの自然言語処理の研究で行われてきた単語をベクトルで表現する手法は「単語の意味は周囲の言語によって形成される」というアイデアに基づいているものがほとんどである。
* これは分布仮説(distributional hypothesis)と呼ばれ、単語自体に意味はなく、その単語のコンテキスト(文脈)によって、単語の意味が形成される、というものである。
* 意味的に同じ単語は同じような文脈で多く出現する。
    * 例)"I drink beer."、"We drink wine."などの文章におけるdrink
* また、以下のような文章があれば、"guzzle"が"drink"と同じような文脈で使われること、意味が近い単語である、ということもわかる。
    * "I guzzle beer."、"We guzzle wine."
* 注目する単語に対するその周囲に存在する単語をコンテキストと呼び、コンテキストに周囲のどこまでの単語を含めるかのサイズをウィンドウサイズ(Window size)と呼ぶ。
* つまりウィンドウサイズが2の場合、左右のそれぞれ2単語がコンテキストとなる。
* ウィンドウサイズは左右均等である必要はなく、どちらか一方だけをコンテキストとする、といったことや分の区切りを考慮したコンテキストを考える、といったこともできる。
* 上記をもとに分布仮説に基づいて単語をベクトルで表現する方法を考える。
* もっとも単純な方法はある単語の周囲にどのような単語がどの程度出現するかをカウントし、それを集計することである。
* まず前述のpreprocess()関数を使って下準備を行う。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    print(corpus)
    # [0,1, 2, 3, 4, 1, 5, 6]

    print(id_to_word)
    # {0:'you', 1:'say', 2:'goodbye', 3:'and', 4:'i', 5:'hello', 6:'.'}
    ```
* 上記より、textは単語数は8、語彙数は7であることがわかる。
* この各単語について、ウィンドウサイズを1とし、コンテキストに含まれる単語の頻度を数えると以下のようになる。
    |         | you   | say   | goodbye | and   | i     | hello | .     |
    |:-------:|:-----:|:-----:|:-------:|:-----:|:-----:|:-----:|:-----:|
    | you     | 0     | 1     | 0       | 0     | 0     | 0     | 0     |
    | say     | 1     | 0     | 1       | 0     | 1     | 1     | 0     |
    | goodbye | 0     | 1     | 0       | 1     | 0     | 0     | 0     |
    | and     | 0     | 0     | 1       | 0     | 1     | 0     | 0     |
    | i       | 0     | 1     | 0       | 1     | 0     | 0     | 0     |
    | hello   | 0     | 1     | 0       | 0     | 0     | 0     | 1     |
    | .       | 0     | 0     | 0       | 0     | 0     | 1     | 0     |
* 1行目の'you'は隣り合っている'say'のみがコンテキストとなるため、1となり、それ以外の単語は0となる。
* これは同時に'you'という単語が[0, 1, 0, 0, 0, 0, 0]というベクトルで表現できることを意味している。
* これを全ての語彙に対して行ない作成されるテーブルを共起行列(co-occurence matrix)と呼ぶ。
    * 共起とは、ある単語がある文中に出現したとき、その文中に別の単語が頻繁に出現すること。
* この共起行列をnp.arrayで定義すると以下のようになる。
    ```python
    C = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0]
    ], dtype=np.int32)
    ```
* この共起行列を使うことで以下の通り、各単語のベクトルを取得することができる。
    ```
    print(C[0]) # 単語IDが0のベクトル
    # [0, 1, 0, 0, 0, 0, 0]

    print(C[4])  # 単語IDが4のベクトル
    # [0, 1, 0, 1, 0, 0, 0]

    print(C[word_to_id['goodbye']]) # 'goodbyeのベクトル'
    # [0, 1, 0, 1, 0, 0, 0]
    ```
* これを踏まえ、コーパスから共起行列を作る関数create_co_matrix()を実装する。
    * create_co_matrix(corpus, vocab_size, window_size=1)
        * corpus：単語IDのリスト
        * vocab_size：語彙数
        * window_size：ウィンドウサイズ
    ```python
    def create_co_matrix(corpus, vocab_size, window_size=1):
        corpus_size = len(corpus)
        co_matrix = np.zeors((vocab_size, vocab_size), dtype=np.int32)

        for idx, word_id in enumerate(corpus):
            for i in range(1, window_size+1):
                left_idx = idx - i
                right_idx = idx + i

                if left_idx >= 0:
                    left_word_id = corpus[left_idx]
                    co_matrix[word_id, left_word_id] += 1
                
                if right_idx < corpus_size:
                    right_word_id = corpus[right_idx]
                    co_matrix[word_id, right_word_id] += 1

        return co_matrix
    ```
* まず語彙数×語彙数の大きさの2次元配列を全要素を0で初期化そて生成する。
* 次にコーパス中の各単語に対して、そのウィンドウに含まれる単語をカウントしていく。
* ただし、コーパスの右端・左端のはみ出しチェックを行い、はみ出していない単語のみカウントする。
* この関数を用いることで任意の大きさのコーパスに対する共起行列を生成することができる。
* 次にベクトル間の類似度を計測する方法を考える。
* 一般的には様々な方法があり、ベクトルの内積やユークリッド距離などが代表的な方法として挙げられる。
* 単語のベクトル表現の類似度に関しては、コサイン類似度(cosine similarity)がよく用いられる。
* コサイン類似度はベクトルx, yに対して、以下の(2.1)で定義される。
    * similarity(x, y) = (x・y) / (||x| ||y||) = (x1y1 + x2y2 +・・・xnyn) / (√x1^2+・・・+xn^2 √y1^2+・・・+yn^2)　(2.1)
        * x = (x1, x2, ... , xn)
        * y = (y1, y2, ... , yn)
* (2.1)は分子にベクトルの内積、分母に各ベクトルのノルムがある。
* ノルムはベクトルの大きさを表したのもので、ここではL2ノルム(ベクトルの各要素の2乗和の平方根)を用いている。
* これはベクトルを正規化して内積を取っていると言える。
* コサイン類似度は直感的には2つのベクトルがどれだけ同じ方向を向いているかを表すものである。
* 2つのベクトルが完全に同じ方向を向いている場合、コサイン類似度は1となり、完全に逆向きだと-1となる。
* これにより、コサイン類似度を実装すると以下のようになる。
    ```python
    def cos_similarity(x, y):
        nx = x / np.sqrt(np.sum(x**2))　＃xの正規化
        ny = y / np.sqrt(np.sum(y**2))　＃yの正規化

        return np.dot(nx, ny)
    ```
* 引数x, yはNumPy配列とし、x, yをそれぞれ正規化した後、両者の内積を求めている。
* 一方、上記の実装だとx, ｙに0ベクトルが渡されてくると、0除算が発生してしまう。
* ここでは、小さな値epsを1e-8とし、引数で指定できるようする。
* 指定されたepsはnx, nyの分母の計算に必ず加算される値として実装する。
* これを踏まえたcos_similarity()の改良版は以下の通りとなる。
    ```python
    def cos_similarity(x, y, eps=1e-8):
        nx = x / np.sqrt(np.sum(x**2)) + eps)　＃xの正規化
        ny = y / np.sqrt(np.sum(y**2)) + eps)　＃yの正規化

        return np.dot(nx, ny)
    ```
* このcos_similarity()改良版を用いて、"you"と"i"の単語ベクトルの類似度を求める実装は以下の通りになる。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess, create_co_matrix, cos_similarity

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[wprd_to_id['you']] # 'you'の単語ベクトル
    c1 = C[wprd_to_id['i']]   # 'i'の単語ベクトル
    print(cos_similarity(c0, c1))
    # 0.7071067691154799
    ```
* "you"と"i"の単語ベクトルの類似度は0.70...と言える。
* コサイン類似度は1から-1までの値を取るので、この値は比較的高い値(=類似正がある)と言える。
* 次にある単語がクエリとして与えられたとき、そのクエリに対して類似した単語を上位から順に表示する関数most_similar()を考える。
    * most_similar()
        * query：クエリ(単語)
        * word_to_id：単語から単語IDへのディクショナリ
        * id_to_word：単語IDから単語へのディクショナリ
        * word_matrix：単語ベクトルをまとめた行列で各行に対応する単語のベクトルが格納(共起行列)
        * top：取得する類似度のランキング
* 実装は以下の通り。
    ```python
    def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
        # Pick up the query
        if query not in word_to_id:
            print('%s is not found' % query)
            return

        print('\n[query] ' + query)
        query_id = word_to_id[query]
        query_vec = word_matrix[quiery_id]

        # calculate cosine similarity
        vocab_size = len(id_to_word)
        similarity = np.zeros(vocab_size)
        for i in range(vocab_size):
            similarity[i] = cos_similarity(word_matrix[i], query_vec)
        
        # Output the value of cosine similarity in descending order 
        count = 0
        for i in (-1 * similarity).argsort():
            if id_to_word[i] == query:
                continue
            print('%s: %s' % (id_to_word[i], similarity[i]))

            count +=1
            if count >= top:
                return
    ```
* まずはじめに渡されてきたqueryがword_to_idのキーに存在するかをチェックする
* 存在する場合は処理をそのqueryを表示し、処理を続行する。
* word_to_idからqueryをキーにその単語のIDを取得する。取得した単語IDを使い、word_matrixからその単語の単語ベクトルを取得し、query_vecに保持する。
* 次にコサイン類似度similarityを計算する。まず、id_to_wordの大きさ分のベクトルを要素0で初期化する。
* cos_similarity()にword_matrixの各行とquery_vecを渡し、コサイン類似度を計算し、各行の値をベクトルsimilarityの要素として保持する。
* 最後にコサイン類似度をその値が高い順に表示するためにargsort()関数を使用する。
* argsort()はNumpy配列の要素を小さい順にソートし、ソート後のインデックスを返す関数である。
    ```python
    >>> x = np.array([100, -20, 2])
    >>> x.argsort()
    # array([1, 2, 0])
    ```
* よって、similarityの各要素に-1をかけ、argsort()関数を適用すると、その返り値は値の大きい要素順のインデックスが変えることになる。
* これによって、for文を回し、id_to_word, similarityを表示する。
* このfor文をtopで渡された回数分回すことで、
* コサイン類似度が高い順にその値と単語を表示することができる。
* 次にこのmost_similar()関数を使用する例は以下の通り。例ではクエリとして'you'を指定する。
    ```python
    import sys
    sys.path.append('..')
    from common.util import preprocess, create_co_matrix, most_similar

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    most_similar('you', word_to_id, id_to_word, C, top=5)
    ```
* これを実行すると以下のような結果が得られる。
* この結果から'you'に近い単語は3つあり、'goodbye', 'i', 'hello'であることがわかる。
    ```
    [query] you
    goodbye: 0.7071067691154799
    i: 0.7071067691154799
    hello: 0.7071067691154799
    say: 0.0
    and: 0.0
    ```
* 'i'は'you'と同じ人称代名詞であるため、コサイン類似度が高いことはよく理解できる。
* 一方、'goodbye'や'hello'のコサイン類似度が高いには説明がつかないが、これはコーパスのサイズが極端に短いことが原因と考えられる。


# カウントベースの手法の改善
* 前述の通り、単語の共起行列を作り、単語をベクトルで表現することができたが、さらに共起行列を改善する。
* 前述の共起行列の要素は2つの単語が共起した回数を表している。
* ただ、これは特に高頻度単語に目を向けると、あまり良い性質を持たないことがわかる。
* 例えば、コーパスにおいて、「the」、「car」の共起を考える。
* その場合、コーパスでは「...the car...」というフレーズは多くみられることが考えられ、共起回数は大きくなる。
* 一方、「car」と「drive」は単語間に明らかに強い関連性があるにもかかわらず、出現回数だけを見ると、「car」は「drive」よりも「the」の方が強い関連性を持ってしまうことになる。
* このような問題を解決するため、相互情報量(PMI：Pointwise Mutual Information)と呼ばれる指標が使われる。
* PMIは確率変数x, yに対して次の式で定義され、値が大きいほどx, yの関連性が高いと言える。
    * PMI(x, y) = log_2 ((P(x, y)) / (P(x)P(y)))　　(2.2)
        * P(x)：xが起きる確率(単語xがコーパスに出現する確率)
        * P(y)：yが起きる確率(単語yがコーパスに出現する確率)
        * P(x, y)：xとyが同時に起きる確率(単語x, yが共起する確率)
* 例えば、10000個の単語からなるコーパスで「the」が100回出現するとする。P('the') = 100/10000 = 0.01となる。
* 次にy = 'car'とし、「the」と「car」が10回共起したとすると、P('the', 'car') = 10/10000 = 0.001となる。
* 共起行列(共起した単語の回数を要素に持つ行列)をC、単語x, yの出現回数及び単語x,yの共起回数をそれぞれC(x), C(y), C(x,y)とし、コーパスに含まれる単語数をNとすると、(2.2)は以下のように表すことができる。
    * PMI(x, y) = log_2 ((P(x, y)) / (P(x)P(y))) = log_2 ((C(x,y)/N) / (C(x)/N・C(y)/N)) = log_2 ((C(x,y)・N) / (C(x)C(y))　　(2.3)
* (2.3)が導出できたことにより、PMIは共起行列から求めることが可能であることがわかる。
* この式を用い、具体的なコーパスに関してPMIを算出する。
* N=10000のコーパスにおいて、「the」「car」「drive」がそれぞれ1000回、20回、10回出現したとする。
* また、「the」と「car」、「car」と「drive」の共起はそれぞれ10回、5回とする。
* この時点で従来の共起行列の観点では、「the」と「car」の方が「car」と「drive」より関連性が高いことが言える。
* 一方、「the」と「car」、「car」と「drive」のPMIを算出すると、以下のようになる。
    * PMI('the', 'car') = log_2 (10・10000)/(1000・20) ≒ 2.32
    * PMI('car', 'drive') = log_2 (5・10000)/(20・10) ≒ 7.97
* 上記より、PMIの観点では、「car」と「drive」の方が「the」と「car」より関連性が高いという結果になる
* これは上述の通り、PMIにより単語単独の出現回数が考慮されてことによると言える。
* 一方、PMIは2つの単語の共起回数が0になる場合、log_2 0 = ∞となってしまう。
* よって、実際には以下で示す、正の相互情報量(PPMI：Positive PMI)が使われる。
    * PPMI(x, y) = max(0, PMI(x,y))　　(2.6)
* これにより、PMIの値がマイナスの場合は、それを0として扱うことができ、単語間の関連度を0以上の実数で表すことができる。
* 共起行列をPPMI行列(要素がPPMI値の行列)に変換する関数を実装すると、以下の通りとなる。
    ```python
    def ppmi(C, verbose=False, eps=1e-8):
        M = np.zeros_like(C, dtype=np.floar32)
        N = np.sum(C)
        S = np.sum(C, axis=0)
        total = C.shape[0] * C.shape[1]
        cnt = 0

        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                pmi = np.log2(C(i, j) * N / (S[j]*S[i]) + eps)
                M[i, j] = max(0, pmi)

                if verbose:
                    cnt+=1
                    if cnt % (total//100) == 0:
                        print('%lf%% done', %(100*cnt/total))
        
        return M
    ```
* この関数を用いて、共起行列をPPMI行列に変換する実装は以下のようになる。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess, create_co_matrix, cos_similarity, ppmi

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)
    W = ppmi(C)

    np.set_printoptions(precision=3)
    print('covariance matrix')
    print(C)
    print('-'*50)
    print('PPMI')
    print(W)
    ```
* 上記を実行すると以下の結果が得られる。
    ```
    covarience matrix
    [[0 1 0 0 0 0 0]
     [1 0 1 0 1 1 0]
     [0 1 0 1 0 0 0]
     [0 0 1 0 1 0 0]
     [0 1 0 1 0 0 0]
     [0 1 0 0 0 0 1]
     [0 0 0 0 0 1 0]]
     --------------------------------------------------
     PPMI
    [[0.     1.807  0.     0.     0.     0.     0.   ]
     [1.807  0.     0.807  0.     0.807  0.807  0.   ]
     [0.     0.807  0.     1.807  0.     0.     0.   ]
     [0.     0.     1.807  0.     1.807  0.     0.   ]
     [0.     0.807  0.     1.807  0.     0.     0.   ]
     [0.     0.807  0.     0.     0.     0.     2.807]
     [0.     0.     0.     0.     0.     2.807  0.   ]]
    ```
* これにより、PPMI行列を取得することができ、共起行列より良い指標の単語ベクトルを取得できたと言える。
* ただ、PPMI行列はこーますの語彙数が増えるにつれ、各単語の単語ベクトルの次元数が増えていくという問題がある。
* つまり、コーパスに含まれる語彙数が10万個の場合、そのベクトルの次元数も同様に10万となることを意味する。
* PPMI行列はその要素の多くが0であり、これはほとんどの要素が重要ではなく、各要素の持つ重要度が低いことを意味する。
* このように0要素が多いベクトルはノイズに弱く、頑健性に乏しいという欠点がある。
* これらの問題に対してはベクトルの次元削減(dimensionality reduction)を行なう。
* ただし、単純に次元を削減するのではなく、重要な情報をできるだけ残したうえで削減する必要がある。
* ここでの次元削減ではデータの分布を見て、重要な軸を見つけることを行なう。
* 例えば、2次元の座標で表されているデータ点を1次元の座標で表すとする。
* その際はデータの広がりを考慮し、その広がりが大きな軸を見つけ、新たな軸を導入する。
* 新しい軸を見つけ、導入する場合、各データ点の値は新しい軸へ射影された値に変換される。
* データの広がりを考慮した軸が取れていれば、1次元の値に変換されても、データの本質的な差異は失わずに捉えたままにできる。
* ベクトル中のほとんどの要素が0である行列は「疎な行列」と呼ばれる。
* 疎な行列から重要な軸を見つけ、より少ない次元で表現し直すことで、次元が少なくほとんどの要素が0でない行列(密な行列)に変換することを考える。
* 次元削減にはいくつかの方法があり、ここでは特異値分解(SVD：Singular Value Decomposition)を用いる。
* SVDでは任意の行列Xを3つの行列U, S, Vの積へと分解する。式で書くと以下の通りとなる。
    * X = U・S・V^T　　(2.7)
* ここで、U及びVは直交行列(=両行列の積が単位行列)、Sは対角行列(対角成分以外の成分が全て0である行列)となる。
* 直交行列Uは単語空間の軸を形成しており、対角行列Sの対角成分には「特異値」が大きい順に並んでいる。
* この特異値は単語空間の軸の重要度とみなすことができる。
* つまり、行列Sの特異値が小さいものは重要度が低いとみなし、それらの要素を削ることで次元を削減し、元の行列の近似した行列を取得する。
* これは単語のPPMI行列から重要度の低い要素を削減した単語ベクトルU'を取得することを意味する。
* SVDの実装はNumpyのlinalgモジュールにあるsvdメソッドで実行できる。
    * linalgはlinear algebra(線形代数)を表す。
* SVDを適用する実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import preprocess, create_co_matrix, ppmi

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    W = ppmi(C)

    U, S, V = np.linalg.svd(W)
    ```
* 上記にて得られる行列UにSVDによって変換された密な単語ベクトル表現が格納されている。
* 共起行列、PPMI行列、SVDで得られる行列を表示すると以下の通り。
    ```python
    print(C[0])　　# 共起行列
    # [0 1 0 0 0 0 0]

    print(W[0])　　# PPMI行列
    # [0.     1.807  0.     0.     0.     0.     0.   ]

    print(U[0])　　# SVD
    # [3.409e-01  -1.110e-16  -1.205e-01  -4.441e-16  0.000e+00  -9.323e-01  2.226e-16]
    ```
* 上記より、疎なベクトルW[0]がSVDによって密なベクトルU[0]に変換されていることがわかる。
* この密なベクトルを次元削減する場合、その先頭の要素を必要な分だけ取り出すことで実現できる。
* 例えば、2次元ベクトルに削減する場合は以下のようになる。
    ```python
    print(U[0, :2])
    # [3.409e-01  -1.110e-16]
    ```
* 次に各単語を2次元ベクトルで表すことにし、それを以下の実装でグラフにプロットしてみる。
    ```python
    for word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    
    plt.scatter(U[:,0], U[:,1], alpha=0.5)
    plt.show()
    ```
* plt.annotate(word, x, y)により、2次元グラフ上の座標(x,y)にwordというテキストを描画する。
* 結果を見ると、「goodbye」と「hello」、「you」と「i」が近い位置にいることがわかる。
* SVDの計算は行列のサイズをNとして場合、O(N^3)というオーダーの計算量になる。
* これは現実的に非常に大きな計算量となるため、通常はTruncated SVDのような高速化手法を用いるのが一般的である。
* Truncated SVDでは特異値が小さなものを切り捨てることで高速化を図っている。
* 上記の結果は、コーパスが小さく結果が正しく出ていない部分もある。
* 以下では、PTBデータセットというより大きなコーパスを使って同じ処理を行って結果を確認する。
* PTBはPenn Treebankと呼ばれるコーパスのことで、提案手法の品質を測定するためのベンチマークとしてよく利用されている。
* PTBコーパスはWord2vecの発明者であるTomas MikolovのWebページで用意されているものでテキストファイルでて供されているが、以下のような前処理が施されている。
    * レアな単語を\<unk>に置換
    * 具体的な数字をNに置換
* 以下ではこれら前処理が施されているテキストデータをPTBコーパスとして使用する。
* PTBコーパスは1つの文が1行ごとに保存されており、ここでは各文を連結したものを1つの大きな時系列データとして扱うことにする。
* 加えて各文の終わりに\<eos>という特殊文字(eos=end of sentence)を挿入する。
* Penn Treebankのデータセットを扱う実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    from dataset import ptb

    corpus, word_to_id, id_to_word = ptb.load_data('train')

    print('corpus size:', len(corpus))
    print('corpus[:30]', corpus[:30])
    print()
    print('id_to_word[0]:', id_to_word[0])
    print('id_to_word[1]:', id_to_word[1])
    print('id_to_word[2]:', id_to_word[2])
    print()
    print('word_to_id['car']:', word_to_id['car'])
    print('word_to_id['happy']:', word_to_id['happy'])
    print('word_to_id['lexus']:', word_to_id['lexus'])
    ```
* 上記を実行すると、以下の結果が得られる。
    ```
    corpus size:929589
    corpus[:30]: [0 1 2 3 4 5 6 7 8...27 28 29]

    id_to_word[0]:aer
    id_to_word[1]:banknote
    id_to_word[2]:berlitz

    word_to_id['car']: 3856
    word_to_id['happy']: 4428
    word_to_id['lexus']: 7426
    ```
* corpusには単語IDのリストが格納され、id_to_word, word_to_idにはそれぞれ、単語IDから単語への変換ディクショナリ、単語から単語IDへの変換ディクショナリが格納される。
* ptb.load_data()ではPTBのデータをloadする。引数で'train', 'test', 'valid'を渡すことにより、それぞれ学習用、テスト用、検証用のデータが指定できる。
* PTBデータセットに対してカウントベースの手法を適用することを考える。
* ここではより高速なSVDを使用するため、sklearnをimportし、randomized_svd()メソッドを使用する。
* randomized_svd()では乱数を使ったTruncated SVDで特異値の大きなものだけに限定して計算し、通常のSVDよりも高速に計算を行うことが可能。
* 実装は以下の通り。
    ```python
    import sys
    sys.path.append('..')
    import numpy as np
    from common.util import most_similar, create_co_matrix, ppmi
    from dataset import ptb

    window_size = 2
    wordvec_size = 100

    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    print('counting co-occurrence...')
    C = create_co_matrix(corpus, vocab_size, window_size=1)
    print('calculating PPMI...')
    W = ppmi(C, verbose=True)

    print('calculating SVD...')
    try:
        # truncated SVD
        from sklearn.utils.extmath import randomized_svd
        U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
    except ImportError:
        # SVD
        U, S, V = np.linalg.svd(W)
    
    word_vecs = U[:, :wordvec_size]

    querys = ['you', 'year', 'car', 'toyota']
    for query in querys:
        most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    ```
* 上記を実行すると、以下の結果が得られる。
    ```
    [query] you
     i: 0.702039909619
     we: 0.699448543998
     've: 0.554828709147
     do: 0.534370693098
     else: 0.512044146526

    [query] year
     month: 0.731561990308
     quarter: 0.658233992457
     last: 0.622425716735
     earlier: 0.607752074689
     next: 0.601592506413
    
    [query] car
     luxury: 0.620933665528
     auto: 0.615559874277
     cars: 0.569818364381
     vehicle: 0.498166879744
     corsica: 0.472616831915
    
    [query] toyota
     motor: 0.738666107068
     nissan: 0.677577542584
     motors: 0.647163210589
     honda: 0.628862370943
     lexus: 0.604740429865
    ```
* a
* ★★～P.89★★


# Method
* P.64：text.lower()
* P.64：text.replace('.', '. ')
* P.64：text,split(' ')
* P.66：[word_to_id[w] for w in words]　※内包表記
* P.69：sys.path.append('..')
* P.72：enumerate(corpus)
* P.75：if query not in word_to_id:
* P.75：x.argsort()
* P.80：np.set_printoptions(precision=3)
* P.85：word_to_id.items()
* P.85：plot.annotate(word, (U[word_id, 0], U[word_id, 1]))
* P.85：plt.scatter(U[:,0], U[:,1], alpha=0.5)






