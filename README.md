# Script de NLP: pedido → SQL

NLP supervisionada em Python com scikit-learn. O script treina usando exemplos de frases em português e queries SQL no arquivo `data/treino.csv`. TF-IDF transforma o texto em características e um classificador SVM aprende a associar essas características às queries. Não usa Ollama, APIs ou modelos de linguagem externos.

## Rodar

```sh
python3 -m pip install -r requirements.txt
python3 nlp_sql.py
```

Ou passe uma frase diretamente:

```sh
python3 nlp_sql.py 'quantos clientes estão cadastrados?'
```

Saída esperada:

```sql
SELECT COUNT(*) AS total_clientes FROM customer;
```

O script retorna a query, sem executá-la. Digite `sair` para encerrar o modo interativo.

## Ensinar novos pedidos

Adicione linhas ao `data/treino.csv`, com as colunas `pedido` e `sql`. Inclua várias formas de pedir a mesma query. Reinicie o script para treinar novamente. A base contém 56 frases para 8 queries: listar clientes/preços, contar clientes, média, soma, maior/menor preço e gastos por cliente.

**Limite deste modelo:** ele classifica a frase e retorna uma das queries ensinadas. Não escreve SQL arbitrário. Suporta um filtro dinâmico de igualdade por `customer_id` conforme descrito abaixo; filtros de nomes, cidades, datas e preços ainda não são suportados. O modelo aprende também com exemplos de conversa em `data/fora_escopo.txt` e recusa previsões com pouca similaridade ou margem pequena entre classes. Esses limites são heurísticos, não garantem detectar todo pedido desconhecido; ainda pode retornar uma query incorreta. Para consultas novas, acrescente exemplos ao treino.

## Esquema dos CSVs de exemplo

- `customer`: `id`, `name`, `email`, `city`, `state`.
- `price`: `id`, `customer_id`, `product`, `amount`, `currency`, `created_at`.

Relação: `price.customer_id = customer.id`. Os CSVs representam dados fictícios; `price` contém compras, com valores em BRL.

## Total geral e total por cliente

- `Quero o total gasto por todos os clientes`: soma geral, `SUM(amount)`.
- `Quero o total gasto por cada cliente`: consulta com `JOIN` e `GROUP BY`.

Para verificar as regressões: `python3 -m unittest -v`. Os testes cobrem estes casos específicos; não representam uma avaliação ampla da precisão do modelo.

## Filtro por cliente

`gastos onde customer_id é 50` retorna:

```sql
SELECT SUM(amount) AS total FROM price WHERE customer_id = 50;
```

O classificador aprende a intenção; um extrator separado identifica o ID numérico e compõe o WHERE. Também aceita `com customer_id = 50` e `para customer_id igual a 50`. O valor pode ser qualquer inteiro não negativo. O script não verifica a existência desse cliente nos CSVs.

`total gasto por cada cliente onde customer_id = 50` preserva o JOIN e aplica o WHERE antes do GROUP BY. `gastos` sozinho solicita mais detalhes. Condições sinalizadas por `onde`, comparadores ou números que o extrator não reconhece são recusadas. Essa proteção é limitada aos marcadores reconhecidos; o modelo continua sem compreender todos os filtros possíveis em linguagem natural.
