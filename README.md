# бот для игры в SC2
магистрская работа

## Установка
1. Установить клиент игры Starcraft 2 в Battle.net
2. установить python (*https://www.python.org/downloads/*)
3. Установить pysc2 (*https://github.com/deepmind/pysc2*)
4. Скачать Мини игры по ссылке где лежит pysc2

### Агенты
1. RandomAgent - агент с автодействиями, взятый в качестве примера с pysc2
2. ScriptAgent MoveToBeacon - агент, который решает задачу MoveToBeacon (просто программирование)
3. Q-Learning Agent MoveToBeacon - агент, который решает задачу MoveToBeacon с помощью метода Q-learning (Reinforsment learning)
4. ScriptAgent CollectMineralShards - агент, который умеет собирать ресрусы
5. Q-learning Agent CollectMineralShards - агент, который умеет собирать ресурсы с помощью метода Q-learning
6. Script Agent DefeatRoaches - агент для мини игры: где есть отряд морпехов и надо победить отряд монстров
7. Q-learning Agent DefeatRoaches - not ready

## Запуск
 *python -m pysc2.bin.agent --map MoveToBeacon --agent moveToBacon.MoveToBaconScriptAgent* - данная команда запустит игру на карте MoveToBeacon с агентом MoveToBaconScriptAgent из файла moveToBacon. Запуск происходит в папке agents
1. python -m pysc2.bin.agent --map MoveToBeacon --agent randomAgent.RandomAgent
2. python -m pysc2.bin.agent --map MoveToBeacon --agent moveToBacon.MoveToBaconScriptAgent
3. python -m pysc2.bin.agent --map MoveToBeacon --agent learningMoveToBeacon.LearningAgent
4. python -m pysc2.bin.agent --map CollectMineralShards --agent CollectMinerals.CollectMineralShards
5. python -m pysc2.bin.agent --map CollectMineralShards --agent learningCollectMinerals.LearningAgent
6. python -m pysc2.bin.agent --map DefeatRoaches --agent defeatRoaches.DefeatRoaches
7. python -m pysc2.bin.agent --map CollectMineralsAndGas --agent learningCollectMineralsAndGas.SmartAgent
8. python -m pysc2.bin.agent --map CollectMineralsAndGas --agent learningCollectMineralsAndGasV2.SmartAgent
9. python -m pysc2.bin.agent --map CollectMineralsAndGas --agent learningCollectMineralsAndGasV3.SmartAgent
10. python -m pysc2.bin.agent --map BuildMarines --agent learningBuildMarines.SmartAgent
11. python -m pysc2.bin.agent --map BuildMarines --agent learningBuildMarinesV2.SmartAgent
12. python -m pysc2.bin.agent --map BuildMarines --agent learningBuildMarinesV3.SmartAgent
13. python -m pysc2.bin.agent --map BuildMarines --agent learningBuildMarinesV4.SmartAgent
14. python -m pysc2.bin.agent --map BuildMarines --agent learningBuildMarinesV5.SmartAgent
15. python RushMarineBot.py - scripted bot with strategy rush marines


