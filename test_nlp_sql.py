import unittest
import sqlite3
from nlp_sql import generate_sql, train


class NLPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train()

    def test_total_geral(self):
        for phrase in ['quero o total gasto por todos os clientes',
                       'quanto todos os clientes gastaram juntos?']:
            with self.subTest(phrase=phrase):
                self.assertEqual(generate_sql(phrase, self.model),
                                 'SELECT SUM(amount) AS total FROM price;')

    def test_total_por_cliente(self):
        sql = generate_sql('Quanto foi gasto por cada um dos clientes?', self.model)
        self.assertIn('JOIN price', sql)
        self.assertIn('GROUP BY c.id', sql)

    def test_conversa_nao_retorna_sql(self):
        for phrase in ['estás bom?', 'k', 'como vai você?', 'me conta uma piada', '']:
            with self.subTest(phrase=phrase), self.assertRaises(ValueError):
                generate_sql(phrase, self.model)

    def test_filtro_customer_id(self):
        sql = generate_sql('gastos onde customer_id é 50', self.model)
        self.assertEqual(sql, 'SELECT SUM(amount) AS total FROM price WHERE customer_id = 50;')
        conn = sqlite3.connect(':memory:')
        try:
            conn.execute('CREATE TABLE price (customer_id INTEGER, amount REAL)')
            conn.executemany('INSERT INTO price VALUES (?, ?)', [(50, 10), (50, 20), (1, 500)])
            self.assertEqual(conn.execute(sql).fetchone()[0], 30)
        finally:
            conn.close()

    def test_filtro_com_join(self):
        sql = generate_sql('total gasto por cada cliente onde customer_id = 50', self.model)
        self.assertIn('JOIN price', sql)
        self.assertIn('WHERE p.customer_id = 50 GROUP BY', sql)

    def test_filtros_nao_suportados(self):
        for phrase in ['gastos', 'gastos onde customer_id > 50',
                       'gastos onde customer_id é 50 ou 60',
                       'gastos onde cidade é Lisboa',
                       'gastos com preço acima de 100']:
            with self.subTest(phrase=phrase), self.assertRaises(ValueError):
                generate_sql(phrase, self.model)

    def test_contagem(self):
        self.assertIn('COUNT(*)', generate_sql('Quantos clientes estão cadastrados?', self.model))


if __name__ == '__main__':
    unittest.main()
