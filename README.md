# Pametni Sistem za Detekciju i Praćenje Uljeza

Ovaj repozitorijum sadrži izvorni kod i prateću dokumentaciju za sistem automatske detekcije, biometrijske verifikacije i inteligentnog praćenja pokretnih objekata (uljeza) pomoću motorizovane PTZ IP kamere. Sistem koristi napredne modele računarskog vida (YOLOv8 i DeepFace) i komunicira sa kamerom preko ONVIF i RTSP protokola.

---

## Projektna Dokumentacija & Razvojni Timeline

Razvoj projekta je tekao inkrementalno kroz modularnu implementaciju i testiranje pojedinačnih funkcionalnosti. Na osnovu strukture i hronologije commit-ova, razvojni proces je mapiran kroz sljedeći timeline:

### Timeline Razvoja Funkcionalnosti

#### Faza 1: Inicijalizacija Video Striminga
* **Primarni fajl:** `picture.py`
* **Opis:** Uspostavljena je bazna komunikacija sa IP kamerom preko **RTSP (Real-Time Streaming Protocol)** protokola koristeći OpenCV biblioteku (`cv2.VideoCapture`). Definisan je prenos preko MJPG kodeka i inicijalna niska rezolucija prikaza (640x360) kako bi se osigurao minimalan latency i stabilan mrežni protok u realnom vremenu.

#### Faza 2: Implementacija ONVIF Kontrolnog Interfejsa
* **Primarni fajl:** `camera.py`
* **Opis:** Razvijen je kompletan, samostalan klijent za interakciju sa PTZ motorima kamere putem **ONVIF standarda** (port 8899). Kreirane su funkcije koje enkapsuliraju XML strukture i šalju ih putem HTTP POST zahtjeva uz HTTP Digest autentifikaciju. Implementirane su ključne komande:
  * `move()` i `relative_move()` za pomjeranje po X (pan) i Y (tilt) osama.
  * `stop()` za momentalno zaustavljanje motora.
  * `set_preset()`, `get_presets()` i `goto_preset()` za upravljanje predefinisanim pozicijama.

#### Faza 3: Validacija i Testiranje Dinamike Motora
* **Primarni fajl:** `controls.py`
* **Opis:** Napravljena je testna skripta za kalibraciju hardvera. Skripta izvršava ciklične sekvence pokreta (pan lijevo/desno, tilt gore/dole) sa preciznim vremenskim pauzama (`time.sleep`). Svrha ove faze bila je mjerenje brzine odziva motora kamere kako bi se spriječilo zagušenje ONVIF kontrolera prebrzim slanjem uzastopnih komandi.

#### Faza 4: Integracija Neuronske Mreže za Detekciju Osoba
* **Primarni fajl:** `test_yolo.py`
* **Opis:** Uveden je **YOLOv8 (You Only Look Once)** model za detekciju objekata u realnom vremenu (korištenjem `person.pt` / `yolov8n-face.pt`). Implementirana je logika za automatsko preusmjeravanje inferencije na GPU (CUDA) ukoliko je dostupan, obezbjeđujući stabilan FPS prilikom iscrtavanja bounding box-ova nad dolaznim frejmovima sa kamere.

#### Faza 5: Finalna Integracija, Biometrija i Inteligentni Tracking
* **Primarni fajl:** `body_tracking.py`
* **Opis:** Centralna produkciona skripta koja povezuje sve prethodne module u jedinstven autonomni sistem bezbjednosti. Dodate su sledeće napredne funkcionalnosti:
  * **Face Recognition:** Integrisana je `DeepFace` biblioteka (sa `Facenet512` modelom i OpenCV backend detektorom) koja vrši analizu lica i poredi ih sa lokalnom bazom u folderu `pictures/`.
  * **Dead Zone Algoritam:** Implementirana je "mrtva zona" od 25% (`DEAD_ZONE = 0.25`). Ako se meta pomjeri van ovog okvira u odnosu na centar slike, sistem automatski računa vektore pomaka i trigeruje ONVIF komande za centriranje kamere.
  * **Sistem Uzbune (Intruder Alert):** U slučaju detekcije nepoznatog lica (`Unknown`), sistem automatski aktivira zvučni alarm (`alarm.mp3`) i pokreće video snimanje incidenta u lokaciju `recordings/` sa kružnim baferom (pre-record) od 5 sekundi.

---

## Korisničko Uputstvo za Podešavanje i Rad sa Kamerom

Ovo uputstvo će vas provesti kroz korake konfiguracije pametne IP kamere preko mobilne aplikacije i pokretanja sistema za automatsko praćenje uljeza.

### 1. Inicijalno Podešavanje Kamere preko Aplikacije "cam720"

Prije pokretanja skripti sa računara, kamera mora biti ispravno konfigurisana unutar lokalne mreže.

1. **Preuzimanje aplikacije:** Instalirajte aplikaciju **cam720** sa Google Play Store-a ili Apple App Store-a na vaš pametni telefon.
2. **Uključivanje uređaja:** Povežite kameru na napajanje i sačekajte oko 30 sekundi da prođe kroz inicijalnu rotaciju i emituje zvučni signal koji označava da je u režimu uparivanja.
3. **Dodavanje kamere:**
   * Otvorite aplikaciju `cam720` i registrujte besplatan nalog.
   * Kliknite na dugme **"+" (Dodaj uređaj)** u gornjem desnom uglu.
   * Izaberite opciju za Wi-Fi povezivanje (Smart Setup) i unesite kredencijale vaše lokalne bežične mreže.
   * Pratite instrukcije na ekranu da završite uparivanje.
4. **Saznavanje i fiksiranje IP adrese kamere:**
   * Nakon što uspješno dodate kameru, uđite u njena podešavanja unutar `cam720` aplikacije i otvorite sekciju **Device Info** (Informacije o uređaju). Tu možete pročitati trenutnu *dinamičku IP adresu* koju joj je ruter dodijelio.
   * Kako se ova adresa ne bi mijenjala pri svakom ponovnom paljenju kamere, uđite u podešavanja vašeg kućnog rutera (preko browsera) i u DHCP sekciji dodijelite kameri **statičku (fiksnu) IP adresu** vezanu za njenu MAC adresu.
5. **Konfiguracija RTSP strima preko Web Browsera:**
   * Otvorite web browser na računaru koji je na istoj mreži i u adresnu traku unesite IP adresu kamere (npr. `http://192.168.50.222`).
   * Prijavite se na administratorski interfejs kamere koristeći fabričke parametre ili kreirajte administratorsku lozinku ako interfejs to zahtijeva.
   * Pronađite sekciju za mrežna podešavanja strima (Stream / Network settings) i potražite stavku za **RTSP**.
   * Podesite opciju RTSP strima tako da radi **bez autentifikacije** (ili postavite Open/Anonymous pristup) kako bi OpenCV skripte mogle nesmetano i bez prekida povlačiti frejmove u realnom vremenu (druga ponuđena opcija).
   * Obavezno kliknite na dugme **Save / Apply** kako bi se sve izmjene trajno sačuvale na internoj memoriji kamere.
6. **Kredencijali za ONVIF:** Korisničko ime za ONVIF kontrolu je `admin`, a lozinka `admin123`.

### 2. Pokretanje Sistema Praćenja

Računar sa kojeg pokrećete kod mora biti povezan na istu lokalnu mrežu (ruter) na kojoj se nalazi kamera.

#### Korak 1: Priprema Baze Poznatih Osoba
Unutar direktorijuma `camera_api/` otvorite folder pod nazivom `pictures/`. Unutar tog foldera ubacite foldere sa jasnim fotografijama lica ukućana. Nazivi foldera će se koristiti kao ime osobe na ekranu (npr. `filip/`, `marija/`).

#### Korak 2: Testiranje Video Striminga
Da biste potvrdili da računar ima pristup video strimu kamere, pokrenite:
```bash
python picture.py
```

Ukoliko se otvori prozor sa nazivom "UDP Stream" koji prikazuje živu sliku sa kamere, mrežna putanja i RTSP link su ispravni. Zatvorite prozor pritiskom na taster **'q'**.

#### Korak 3: Aktivacija Glavnog Tracking Sistema

Za pokretanje autonomnog praćenja i detekcije uljeza, pokrenite centralnu skriptu:

```bash
python body_tracking.py
```

### 3. Kako Sistem Reaguje u Radu?

* **Zeleni Okvir (Poznato lice):** Ukoliko se osoba pojavi ispred kamere i sistem je prepozna na osnovu slike iz foldera `pictures/`, oko njenog lica će se iscrtati zeleni okvir sa njenim imenom. Kamera će je pratiti ako izađe iz centralne zone.
* **Crveni Okvir & Alarm (Uljez):** Ukoliko sistem detektuje osobu čije se lice ne nalazi u bazi podataka, automatski se preduzimaju sljedeće mjere zaštite:
1. Oko lica se iscrtava crveni okvir sa oznakom `Unknown`.
2. Pokreće se zvučna repodukcija fajla `alarm.mp3` radi odvraćanja uljeza.
3. Sistem automatski snima video zapis incidenta u folder `recordings/` u formatu `intruder_YYYYMMDD_HHMMSS.avi`. Snimak uključuje i prethodnih 5 sekundi prije same detekcije zahvaljujući implementiranom kružnom baferu.
* **Mobilna Obavještenja preko ntfy.sh servisa (Instant Push):** Istovremeno sa pokretanjem video snimanja i aktivacijom alarma, sistem šalje instant push notifikaciju na vaš pametni telefon kako biste odmah znali da je bezbjednost ugrožena, čak i ako niste u blizini računara.
  
  **Kako se podešava i kako radi?**
  1. **Preuzimanje aplikacije:** Skinite aplikaciju **ntfy** sa Google Play Store-a (ili Apple App Store-a) na vaš telefon.
  2. **Pretplata na Topic (Kanal):** Unutar aplikacije kliknite na dugme za dodavanje novog kanala (`+`) i unesite jedinstveno ime kanala (tzv. *topic*) koji želite pratiti. Naziv kanala mora da se poklapa sa onim koji je definisan u kodu skripte.
  3. **Format i struktura notifikacije:** Kada sistem detektuje nepoznato lice, on šalje HTTP POST zahtjev na `https://ntfy.sh/<vas_topic>`. Notifikacija na vaš telefon stiže u sljedećem formatu:
     * **Naslov (Title):** `🚨📷 ALARM - LOCKINATOR`
     * **Poruka (Message):** `Intruder detected! YYYY-MM-DD hh:mm:ss`
     * **Prioritet (Priority):** Postavljen je na najviši nivo (`5` ili `max`), što znači da će telefon glasno zvoniti i zaobići "Do Not Disturb" (Ne uznemiravaj) režim ukoliko mu to dozvolite u postavkama aplikacije.

### Važne Napomene za Bezbjedan Rad

* **Prekid rada:** Da biste bezbjedno ugasili sistem i oslobodili resurse kamere, pritisnite taster **'q'** na tastaturi dok vam je fokusiran prozor sa video prikazom.
* **Lažne uzbune:** Osigurajte da su fotografije u folderu `pictures/` dobro osvijetljene i slikane sprijeda kako sistem ne bi greškom aktivirao alarm za poznate osobe u uslovima slabijeg osvjetljenja.