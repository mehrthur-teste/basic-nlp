"""NLP supervisionada: aprende a associar pedidos em português a queries SQL."""
import argparse
import csv
import re
import unicodedata
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.svm import LinearSVC
except ImportError:
    raise SystemExit('Instale as dependências: python3 -m pip install -r requirements.txt')

EXAMPLES = Path(__file__).resolve().parent / 'data' / 'treino.csv'
OUT_OF_SCOPE = '__FORA_ESCOPO__'


def train(path=EXAMPLES):
    with Path(path).open(encoding='utf-8', newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows or any(not row.get('pedido', '').strip() or not row.get('sql', '').strip() for row in rows):
        raise ValueError('O CSV de treino precisa das colunas pedido e sql, com valores preenchidos.')
    if len({row['sql'] for row in rows}) < 2:
        raise ValueError('Inclua exemplos de pelo menos duas queries diferentes.')
    negatives = EXAMPLES.with_name('fora_escopo.txt').read_text(encoding='utf-8').splitlines()
    texts = [row['pedido'] for row in rows] + negatives
    labels = [row['sql'] for row in rows] + [OUT_OF_SCOPE] * len(negatives)
    model = Pipeline([
        ('texto', FeatureUnion([
            ('palavras', TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 2), sublinear_tf=True)),
            ('caracteres', TfidfVectorizer(strip_accents='unicode', analyzer='char_wb', ngram_range=(3, 5))),
        ])),
        ('classificador', LinearSVC(random_state=0)),
    ])
    model.fit(texts, labels)
    model.known_requests_ = model.named_steps['texto'].transform([row['pedido'] for row in rows])
    return model


def extract_filter(request):
    text = ''.join(c for c in unicodedata.normalize('NFD', request.lower())
                   if unicodedata.category(c) != 'Mn').strip().rstrip('?.!')
    pattern = r"\b(?:onde|com|para)\s+(?:o\s+)?customer_id\s*(?:e|=|igual a|igual)\s*(\d+)\s*$"
    match = re.search(pattern, text)
    if match:
        base = text[:match.start()].strip()
        # Outro filtro no trecho restante não pode ser silenciosamente descartado.
        if re.search(r'\b(onde|customer_id)\b|[<>=]|\d', base):
            raise ValueError('Ainda não sei interpretar essa combinação de filtros.')
        return base, int(match.group(1))
    if re.search(r'\b(onde|customer_id|cujo|cuja|apenas|somente|com|para|acima|abaixo|antes|depois|entre)\b|[<>=]|\d', text):
        raise ValueError('Filtro não suportado. Use, por exemplo: total gasto onde customer_id é 50.')
    return text, None


def apply_customer_filter(sql, customer_id):
    # Os templates padrão não possuem WHERE. Templates personalizados com WHERE
    # exigem um compositor de SQL mais completo; recuse para preservar a semântica.
    if re.search(r'\bWHERE\b', sql, re.I):
        raise ValueError('Este template já contém um filtro e não pode receber outro automaticamente.')
    if re.search(r'\bJOIN price AS p\b', sql, re.I):
        column = 'p.customer_id'
    elif re.search(r'\bFROM price\b', sql, re.I):
        column = 'customer_id'
    elif re.search(r'\bFROM customer\b', sql, re.I):
        column = 'id'
    else:
        raise ValueError('Não reconheci a tabela deste template para aplicar o filtro.')
    query = sql.rstrip().rstrip(';')
    split = re.search(r'\b(GROUP BY|ORDER BY|LIMIT|HAVING)\b', query, re.I)
    position = split.start() if split else len(query)
    return (query[:position].rstrip() + f' WHERE {column} = {customer_id} '
            + query[position:]).rstrip() + ';'


def generate_sql(request, model):
    if not request.strip():
        raise ValueError('Digite um pedido.')
    base, customer_id = extract_filter(request)
    if len(base.split()) < 2 and customer_id is None:
        raise ValueError('Pedido muito curto. Especifique, por exemplo: total gasto ou gastos por cliente.')
    if not base:
        raise ValueError('Informe o que deseja consultar além do filtro.')
    features = model.named_steps['texto'].transform([base])
    if features.nnz == 0:
        raise ValueError('Pedido desconhecido. Acrescente exemplos ao data/treino.csv.')
    prediction = str(model.predict([base])[0])
    similarity = cosine_similarity(features, model.known_requests_).max()
    scores = sorted(model.decision_function([base])[0])
    # Heurísticas de abstenção, não probabilidades calibradas.
    if prediction == OUT_OF_SCOPE or similarity < 0.25:
        raise ValueError('Não reconheci uma consulta suportada nesse pedido. Reformule ou adicione exemplos ao treino.')
    if scores[-1] - scores[-2] < 0.15:
        raise ValueError('Pedido ambíguo para o modelo. Reformule com mais detalhes.')
    if customer_id is not None:
        return apply_customer_filter(prediction, customer_id)
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pedido', nargs='?', help='Pedido em linguagem natural')
    parser.add_argument('--treino', type=Path, default=EXAMPLES, help='CSV com exemplos pedido,sql')
    args = parser.parse_args()
    try:
        model = train(args.treino)
    except (OSError, ValueError) as error:
        parser.exit(1, f'Erro ao treinar: {error}\n')
    if args.pedido:
        try:
            print(generate_sql(args.pedido, model))
        except ValueError as error:
            parser.exit(1, f'Erro: {error}\n')
        return
    print('NLP treinada. Escreva seu pedido ou digite sair.')
    while True:
        try:
            request = input('\nPedido: ').strip()
            if request.lower() in {'sair', 'exit', 'quit'}:
                break
            if request:
                print(generate_sql(request, model))
        except (KeyboardInterrupt, EOFError):
            break
        except ValueError as error:
            print(f'Erro: {error}')


if __name__ == '__main__':
    main()
