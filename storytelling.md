Ato 1: A Bagagem Invisível (Antes do Primeiro Dia) 
A história começa demonstrando que a empresa já lida com um problema invisível 
desde o processo seletivo. 
● A Carga Pré-existente: Nem todo problema nasce na empresa. Utilizando os 
dados, vocês cruzam a coluna de histórico familiar (family_history) com a 
busca por tratamento (treatment) para mostrar que muitos funcionários já 
entram com uma bagagem psicológica. 
● A Omissão na Entrevista: Por precisarem do emprego, as pessoas mentem 
ou omitem seu estado real. Cruzando mental_health_interview e 
phys_health_interview, o gráfico provará que os candidatos se sentem muito 
mais confortáveis em relatar problemas físicos do que mentais aos potenciais 
empregadores. 
● O Gancho: Se os funcionários já chegam com problemas e têm medo de falar 
sobre eles na entrevista, o RH contrata "no escuro". 
Ato 2: O Ecossistema Corporativo (Como o ambiente molda o funcionário) 
Uma vez dentro da empresa, como a rotina afeta a mente? 
● O Paradoxo do Trabalho Remoto: O professor sugeriu agrupar e separar 
quem trabalha remoto (remote_work) de quem atua presencialmente. Será 
que o trabalho remoto traz mais saúde mental, ou o isolamento prejudica? 
Vamos cruzar isso com a facilidade de conversar com colegas (coworkers) e 
supervisores (supervisor) para provar se a distância ajuda ou agrava a 
situação. 
● O Tamanho da Empresa: Usando a coluna no_employees, vocês mostram as 
diferenças entre Startups (1-25 funcionários) e grandes corporações (mais de 
1000) na facilidade de tirar licenças médicas (leave). 
Ato 3: A Falha do Sistema e a Cultura do Medo (O Clímax) 
Aqui é o ponto alto sugerido pelo professor: não basta a empresa dar benefícios se 
a cultura for punitiva. 
● A Barreira do Anonimato: Vocês filtram os casos onde o anonimato não é 
garantido (anonymity = No) e comparam com a busca por tratamento. Isso 
provará que a falta de sigilo afasta as pessoas da ajuda, mesmo que a 
empresa ofereça plano de saúde. 
● A Cultura da Punição: Cruzando a coluna de consequências observadas 
(obs_consequence), vocês mostram que ver colegas sofrerem impactos 
negativos por problemas mentais faz com que a maioria tenda a reprimir suas 
próprias dores. 
Ato 4: O Gran Finale (A Proposta de Machine Learning) 
A conclusão da história é a cereja do bolo. Já que o sistema falha, o RH é cego e as 
pessoas têm medo, vocês apresentam o modelo preditivo como a verdadeira 
solução de valor e possibilidade de extensão do trabalho. 
● O Problema a ser Previsto (O Target do Modelo): Como não temos uma 
coluna chamada "Burnout", o modelo de Classificação usará a coluna 
work_interfere (Se a condição mental interfere no trabalho) como a variável 
alvo a ser prevista. Prever se a interferência será "Often/Sometimes" 
(Frequente) ou "Rarely/Never" (Rara) é a métrica perfeita para prever a 
queda de produtividade/Burnout. 
● As Features de Aprendizado: O algoritmo não vai usar dados médicos (que 
são difíceis de obter e complexos), mas sim variáveis comportamentais e do 
ambiente de trabalho: remote_work, anonymity, leave (dificuldade de licença), 
no_employees e supervisor. 
● A Proposta de Valor Final (Discurso de Fechamento): "Nossa análise provou 
que benefícios não funcionam sem cultura e sigilo. Portanto, criamos um 
modelo que analisa o formato de trabalho e o ambiente. Se o modelo 
classificar que um grupo de funcionários sob certas condições tem alta 
possibilidade de interferência no trabalho (work_interfere), a empresa não 
precisa esperar eles pedirem socorro. Ela atua preventivamente mudando a 
cultura daquele setor, ajustando o trabalho remoto ou reforçando as políticas 
de sigilo de forma anônima e proativa." 