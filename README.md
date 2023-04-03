# regens-take-home

A feladat során megvalósításra került minden követelmény, valamint a Docker konténerizáció is.

## Döntések

A megoldás során nem használtam virtual environmentet, mivel a megoldás Docker konténerizált, így a dependency-elkülönítést ez önmagában megoldja, és felesleges overheadet rakna erre a use-casere, de kétségkívül megvalósítható, ha átmásoljuk az image-be a venv mappát és, entrypointnak a venv-et használjuk.

Az argumentumok parancssori fogadására a Python beépített argparse libraryjét alkalmaztam. A konfigurálható paraméterek a leírásban kértek, valamint az összes, amit az SKlearn SVR modellje támogat, default értéknek az eredeti defaultot adtam meg. Kötelezően nem kell megadni semmilyen argumentumot, defaultokkal kezeltem ezeket az eseteket, alapesetben a megadott dataseten tanít minden feature-rel együtt az SVR alapbeállításaival, majd a /results mappába menti az elkészült ábrákat.

## Eredmények

Mivel regression modellről beszélünk, ezért a klasszifikációnál megszokott confusion mátrix itt nem működőképes, ezért azt igyekeztem vizualizálni, hogy egy adott predikció mennyire állt közel a valós számértékhez. Ez az eredményt annyiban torzította, hogy az eredeti targetünk integereket tartalmaz, viszont float-ot kapunk a modelltől, így nem teljesen egyértelműen megállapítható a modell teljesítménye első ránézésre, de az SVM-eknél várt diagonális vonal itt is megjelenik. A hisztogram vödrökbe sorolja az egyes tippeket, és azt mutatja meg, hogy eredetileg az ugyanolyan labelből mennyi volt. Ez regressziós modellként nem teljesen állja meg a helyét, mint evaluációs eszköz, mivel fontos információ veszik el a kerekítéssel, de ránézésre egy nagyjáboli képet adhat arról, hogy hogyan teljesít a modell.
A mean squared error és az r2 score mindegyik plot-on szerepel, ez jó kiindulópont lehet az evaluációra.
A plotokról minta található a results mappában, amik a fejlesztés során lettek legenerálva.

## Docker

A Docker konténer elfogad argumentumokat, amennyiben rendelkezésünkre áll az image, (https://hub.docker.com/r/balintsipos/regensdocker) (sudo) docker run regensdocker \<arguments\> paranccsal futtatható.
