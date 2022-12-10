### Input

As séries temporais utilizadas neste trabalho são coletas de tensão e corrente das três fases do motor de giro do torno. As séries temporais foram registradas a partir de operações em um sistema real de usinagem presente no Serviço Nacional de Aprendizagem Industrial (SENAI) em colaboração com o Laboratório de Automação Industrial e Inteligência Computacional (LAIIC). O procedimento utilizado neste trabalho é regulamentado pela ISO 3685/1993. Consiste em executar sucessivos passes de usinagens com a ferramenta e examinar em intervalos regulares a sua condição de desgaste. Este processo é repetido até que o desgaste da ferramenta atinja um limite pré-estabelecido. Considerando as diretrizes técnicas da ISO 3685/1993, o limite adotado foi o comprimento máximo do desgaste de flanco de 0,6 mm. Assim, o desgaste de flanco superior ao limite estabelecido foi considerado uma condição inadequada. Além disso, as condições de usinagem foram: profundidade de corte de 0,5 mm, velocidade de corte de 120 m/min e avanço de 0,156 mm/rot.


 - Cada série temporal tem 3584 amostras.
 - Cada série temporal compreende 6 variáveis (Tensão A, Tensão B, Tensão C, Corrente A, Corrente B, and Corrente C)
 - Existem 785 séries temporais no total.
 - 552 estão relacionadas a uma ferramenta em condição adequada, 80 relacionadas com uma ferramenta em condição intermediária e 153 a uma em condição inadequada.
   - Target = 0: Condição Adequada
   - Target = 1: Condição Intermediária
   - Target = 2: Condição Inadequada


Os dados de entrada seguem a seguinte tabela:


| ID  | Time_ID | Tensão  A | Tensão  B | Tensão  C | Corrente A | Corrente B | Corrente C | Target |
|-----|---------|-----------|-----------|-----------|------------|------------|------------|--------|
| 1   | 1       |  1.91     | -0.74     | -0.50     | -0.52      |  0.43      | -1.03      |      0 |
| 1   | 2       |  1.06     | -1.76     | -0.37     | -1.12      |  0.45      |  1.17      |      0 |
| ... | ...     | ...       | ...       | ...       | ...        | ...        |  ...       | ...    |
| 1   | 3584    |  0.48     | -0.22     | -0.24     | -0.33      |  0.95      | -1.69      |      0 |
| 2   | 1       |  0.78     | -0.22     | -0.36     |  0.16      | -0.06      |  1.45      |      0 |
| 2   | 2       | -0.21     | -0.74     | -0.24     | -1.18      |  1.09      |  0.46      |      0 |
| ... | ...     | ...       | ...       | ...       | ...        | ...        | ...        | ...    |
| 2   | 3584    | -1.18     |  0.79     |  0.24     |  1.65      | -0.09      | -0.09      |      0 |
| ... | ...     | ...       | ...       | ...       | ...        | ...        | ...        | ...    |
| 785 | 1       | -0.85     |  0.28     |  2.81     | -0.74      |  2.15      |  0.07      |      2 |
| 785 | 2       | -1.18     |  0.79     | -0.37     |  0.56      | -1.21      |  0.64      |      2 |
| ... | ...     | ...       | ...       | ...       | ...        | ...        | ...        | ...    |
| 785 | 3584    | -0.83     |  1.82     | -0.56     |  1.52      |  0.60      | -0.97      |      2 |
