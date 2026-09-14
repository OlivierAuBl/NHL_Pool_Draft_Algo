# Description des fichiers csv
## fichier skater
### Structure
NHLID, NHL ID (pour matcher avec l'API NHL)
FullName, nom du joueur
Team, équipe 
Pos, F,D,T,G
Age, age en année
LY_GP, statistique de GP de l'année précédente
Pts, statistique de point de l'année précédente
DFO_Ligne, ligne donnée par DFO 
DFO_PP, ligne de PP donnée par DFO
DFO_Injury, injury information DFO
DFO_Update, date update
ScottCullen, prediction centrale en pts de [source=scott cullen]
CBS, prediction centrale de [source]
LineupExpert, prediction centrale de [source]
ESPN, prediction centrale de  [source]
PoolPro, prediction centrale de [source]
HockeyMag, prediction centrale de [source]
Hastag, prediction centrale de [source]
PoolExpert, prediction centrale de [source]
PoolPro_Top prediction plafond de Poolpro

### Meaningful lines
NHLID,FullName,Team,Pos,Age,LY_GP,Pts,DFO_Ligne,DFO_PP,DFO_Injury,DFO_Update,ScottCullen,CBS,LineupExpert,ESPN,PoolPro,HockeyMag,Hastag,PoolExpert,PoolPro_Top
8476453,Nikita Kucherov,TBL,F,32,78,121,F1,PP1,,2025-09-26,125,137.17,119,#N/A,116,129,114,123,150
8477492,Nathan MacKinnon,COL,F,30,79,116,F1,PP1,,2025-09-26,122,124.75,111,119,120,115,115,118,150
8478402,Connor McDavid,EDM,F,28,67,100,F1,PP1,,2025-09-29,121,123.87,125,128,118,133,127,127,165
8477934,Leon Draisaitl,EDM,F,29,71,106,F1,PP1,,2025-09-29,111,108.79,106,108,114,115,110,113,140
8476853,Morgan Rielly,TOR,D,31,82,41,D2,PP1,,2025-09-29,45,52.74,49,47,43,47,50,46,75
8481554,Kaapo Kakko,SEA,F,24,79,44,-,No,OUT,-,44,#N/A,52,56,48,46,42,49,55
8478458,Jack Roslovic,CAR*,F,28,81,39,#N/A,#N/A,#N/A,#N/A,38,#N/A,33,#N/A,34,0,41,37,45
8481721,Arseny Gritsyuk,NJD,0,24,0,0,F4,PP2,,2025-09-26,38,#N/A,42,#N/A,0,41,48,24,0
8477444,Andre Burakovsky,CHI,F,30,79,37,F1,PP2,,2025-09-26,38,#N/A,36,#N/A,39,49,35,33,60

## fichier team
### Structure
Team, 3letter team
LY W, nb w last year
LY L, nb l last year
LY OTL, nb otl last year
LY Pts, nb pts last year
ScottCullen, Projection de [Source=nom_colonne]
CBS,Projection de [Source=nom_colonne]
LineupExpert,Projection de [Source=nom_colonne]
ESPN,Projection de [Source=nom_colonne]
PoolPro,Projection de [Source=nom_colonne]
PoolExpert,Projection de [Source=nom_colonne]
Hastag, Projection de [Source=nom_colonne]

### Precision
je ne me souviens plus, mais possiblement que j'ai du sortir les résultats via les stats des gardiens et normalisé les points avec 82 games pour avoir qqc de cohérent et comparable

### Meaningful Lines
Team,LY W,LY L,LY OTL,LY Pts,ScottCullen,CBS,LineupExpert,ESPN,PoolPro,PoolExpert,Hastag
ANA,35,37,10,80,81.00734523,80.64577398,75.6118068,85.99966622,94.43528586,80.55652174,72.00511073
BOS,33,39,10,76,85.05771249,80.64577398,79.7549195,80.20193591,81.23465451,74.51478261,87.78705281
BUF,36,39,7,79,87.08289612,83.39506173,84.93381038,80.20193591,86.31182041,83.5773913,85.81431005
CAR,47,30,5,99,107.3347324,107.2222222,101.5062612,109.1905874,94.43528586,106.7373913,96.66439523
CBJ,40,33,9,89,79.99475341,91.64292498,85.96958855,91.79739653,86.31182041,93.64695652,83.84156729

# fichier goalie
### Structure

NHLID	,	NHL ID (pour matcher avec l'API NHL)
FullName	,	nom
Team	,	3 letter team
Pos	,	G
Age	,	age en anne
LY_GP	,	statistique de GP de l'année précédente
LY Pts	,	Calcul des points (3Pts SO, 1 OTL, 2W, pas but ni passe)
LY Win	,	Nb win last year
LY OTL	,	NB OTL last Year
LY SO	,	NB SO last year
DFO_ligne	,	DFO ligne info (G1-g2-g3)
DFO_Injury	,	injury info DFO
Update	,	DFO Update date
        
calc_Hashtag_Score	,	"calculated" ou "projection source". Les calculs sont des histoire de type median par exemple, OTL aussi par exemple mais pas forcément précisé. Par fois un tayx de SO par goalie et nombre de match joué, idem pour le taux de OTL parfois. Ce qui est sur c'est W
Projections_Hashtag_Win	,	Hashtag = source
Projections_Hashtag_SO	,	
Projections_Hashtag_OTL	,	
calc_HockeyMag_Score	,	
Projections_HockeyMag_Win	,	
Projections_HockeyMag_SO	,	
calc_HockeyMag_OTL	,	
calc_LineupExpert_Score	,	
Projections_LineupExpert_Win	,	
calc_LineupExpert_SO	,	
Projections_LineupExpert_OTL	,	
calc_PoolPro_Score	,	
Projections_PoolPro_Win	,	
Projections_PoolPro_SO	,	
calc_PoolPro_OTL	,	
calc_PoolPro_Top_Score	,	
Projections_PoolPro_Top_Win	,	
calc_PoolPro_Top_SO	,	
calc_PoolPro_Top_OTL	,	
calc_ScottCullen_Score	,	
Projections_ScottCullen_Win	,	
Projections_ScottCullen_SO	,	
Projections_ScottCullen_OTL	,	
calc_ESPN_Score	,	
Projections_ESPN_Win	,	
Projections_ESPN_SO	,	
Projections_ESPN_OTL	,	
calc_CBS_Score	,	
Projections_CBS_Win	,	
Projections_CBS_SO	,	
calc_CBS_OTL	,	
calc_PoolExpert_Score	,	
Projections_PoolExpert_Win	,	
Projections_PoolExpert_SO	,	
Projections_PoolExpert_OTL	,	


### Meaningful Lines
NHLID,FullName,Team,Pos,Age,LY_GP,LY Pts,LY Win,LY OTL,LY SO,DFO_ligne,DFO_Injury,Update,calc_Hashtag_Score,Projections_Hashtag_Win,Projections_Hashtag_SO,Projections_Hashtag_OTL,calc_HockeyMag_Score,Projections_HockeyMag_Win,Projections_HockeyMag_SO,calc_HockeyMag_OTL,calc_LineupExpert_Score,Projections_LineupExpert_Win,calc_LineupExpert_SO,Projections_LineupExpert_OTL,calc_PoolPro_Score,Projections_PoolPro_Win,Projections_PoolPro_SO,calc_PoolPro_OTL,calc_PoolPro_Top_Score,Projections_PoolPro_Top_Win,calc_PoolPro_Top_SO,calc_PoolPro_Top_OTL,calc_ScottCullen_Score,Projections_ScottCullen_Win,Projections_ScottCullen_SO,Projections_ScottCullen_OTL,calc_ESPN_Score,Projections_ESPN_Win,Projections_ESPN_SO,Projections_ESPN_OTL,calc_CBS_Score,Projections_CBS_Win,Projections_CBS_SO,calc_CBS_OTL,calc_PoolExpert_Score,Projections_PoolExpert_Win,Projections_PoolExpert_SO,Projections_PoolExpert_OTL,,,,,,,,
8476945,Connor Hellebuyck,WPG,G,32,63,121,47,3,8,G1,DTD,2025-09-29,101,40,6,3,103,40,6,5,99.5,39,6,5,98,39,5,5,110,45,6,2,89,35,4,7,103,40,6,5,96,40,3,5,98,43,6,2,,,,,,,,
8476883,Andrei Vasilevskiy,TBL,G,31,63,99,38,5,6,G1,,2025-09-26,86,35,4,4,93.5,37,5,4.5,85,34,4,5,95.5,38,5,4.5,110,45,6,2,85,35,3,6,90,37,4,4,90,38,3,4.5,96,37,6,4,,,,,,,,
8479979,Jake Oettinger,DAL,G,26,58,82,36,4,2,G1,,2025-09-26,87,36,3,6,96,39,4,6,88,35,4,6,90,36,4,6,108,45,5,3,80,31,4,6,92,36,5,5,85,35,3,6,84,36,3,3,,,,,,,,
8477480,Eric Comrie,WPG,G,30,20,25,9,1,2,G2,,2025-09-29,29,11,2,1,1.5,0,0,1.5,20.485,9,0,1,19.5,9,0,1.5,50,25,0,0,29,12,1,2,28,13,0,2,25,10,1,1.5,17,7,1,0,,,,,,,,
8476914,Joonas Korpisalo,BOS,G,31,27,34,11,3,3,G2,,2025-09-26,36,15,1,3,2.5,0,0,2.5,25,10,1,2,27.5,11,1,2.5,1,0,0,1,30,11,1,5,25,10,1,2,28,11,1,2.5,27,11,1,2,,,,,,,,
8475831,Philipp Grubauer,SEA,G,33,26,17,8,1,0,G2,,2025-09-27,36,15,1,3,#N/A,0,0,#N/A,#N/A,13,#N/A,3,#N/A,7,0,#N/A,#N/A,0,0,#N/A,27,11,1,2,#N/A,#N/A,#N/A,#N/A,#N/A,13,1,#N/A,23,11,0,1,,,,,,,,
8477992,Jonas Johansson,TBL,G,30,19,24,9,3,1,G2,,2025-09-26,26,10,1,3,#N/A,0,0,#N/A,#N/A,8,#N/A,2,#N/A,9,0,#N/A,#N/A,0,0,#N/A,23,10,0,3,#N/A,#N/A,#N/A,#N/A,#N/A,9,1,#N/A,24,9,1,3,,,,,,,,
8482821,Arvid Soderblom,CHI,G,26,36,27,10,7,0,G2,,2025-09-26,15,6,0,3,#N/A,0,0,#N/A,#N/A,8,#N/A,4,#N/A,3,0,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,#N/A,18,8,0,2,,,,,,,,
