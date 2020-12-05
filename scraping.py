import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

#home_team, away_team, home_team_win, ht_worth, at_worth, ht_table, at_table, ht_fs, aw_fs, curr_matchday
def season(num):
	max_matchday=35
	curr_matchday=1 #changes in loop
	final_n,final_p,final_m,final_w,final_f,final_md=[],[],[],[],[],[]

	headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
	worths = requests.get(f'https://www.transfermarkt.de/bundesliga/startseite/wettbewerb/L1/plus/saison_id?saison_id={str(num)}', headers=headers)
	soup = BeautifulSoup(worths.content, 'html.parser')
	names_results = soup.findAll('td', class_='hauptlink no-border-links hide-for-small hide-for-pad')#.find('a', class_='vereinprofil_tooltip tooltipstered')
	worths_results = soup.findAll('td', class_='rechts hide-for-small hide-for-pad')
	worths_results = worths_results[2:]
	del worths_results[::2]

	teams = {
	"Hertha BSC": [1],
	"Eintracht Frankfurt": [2],
	"Borussia Mönchengladbach": [3],
	"1.FC Union Berlin": [4],
	"Bayer 04 Leverkusen": [5],
	"RasenBallsport Leipzig": [6],
	"1.FSV Mainz 05": [7],
	"VfB Stuttgart": [8],
	"FC Augsburg": [9],
	"Borussia Dortmund": [10],
	"Arminia Bielefeld": [11],
	"1.FC Köln": [12],
	"FC Schalke 04": [13],
	"SV Werder Bremen": [14],
	"TSG 1899 Hoffenheim": [15],
	"FC Bayern München": [16],
	"SC Freiburg": [17],
	"VfL Wolfsburg": [18],
	"Fortuna Düsseldorf": [19],
	"SC Paderborn 07": [20],
	"Hannover 96": [21],
	"1.FC Nürnberg": [22]
	}

	def replace_names(name):
		if name == "TSG Hoffenheim":
			name_result = "TSG 1899 Hoffenheim"
		elif name == "Werder Bremen":
			name_result = "SV Werder Bremen"
		elif name == "E. Frankfurt":
			name_result = "Eintracht Frankfurt"
		elif name == "F. Düsseldorf":
			name_result = "Fortuna Düsseldorf"
		elif name == "Bor. M'gladbach":
			name_result = "Borussia Mönchengladbach"
		elif name == "Bay. Leverkusen":
			name_result = "Bayer 04 Leverkusen"
		elif name == "Leverkusen":
			name_result = "Bayer 04 Leverkusen"
		elif name == "Bor. Dortmund":
			name_result = "Borussia Dortmund"
		elif name == "Bayern München":
			name_result = "FC Bayern München"
		elif name == "RB Leipzig":
			name_result = "RasenBallsport Leipzig"
		elif name == "SC Paderborn":
			name_result = "SC Paderborn 07"
		elif name == "Union Berlin":
			name_result = "1.FC Union Berlin"
		else:
			name_result = name
		return name_result


	for c,i in enumerate(names_results):
		b = worths_results[c].text.strip()
		if 'Mio.' in b:
			b=float((b[:-7]).replace(',','.'))
		elif 'Tsd.' in b:
			b=float(str((float(b[:-7])/1000)).replace(',','.'))
		else:
			pass
		teams[i.text.strip()].append(b)

	forms=[]
	def init_forms():
		for i in range(1,23):
			forms.append(form(i))

	def spec_form(all):
		re = 0
		for i in all:
			if str(curr_matchday-1)==i[2]:
				if all.index(i)>=4:
					sub=all[all.index(i)-4:all.index(i)]
				elif all.index(i)==3:
					sub=all[all.index(i)-3:all.index(i)]
				elif all.index(i)==2:
					sub=all[all.index(i)-2:all.index(i)]
				elif all.index(i)==1:
					sub=all[all.index(i)-1:all.index(i)]
				elif all.index(i)==0:
					sub=all[all.index(i)]
				else:
					pass

				for x in sub:
					s=x[-1]
					if 'n.V.' in s: s=s[:-4]
					if 'n.E.' in s: s=s[:-4]
					ls=int(s[:s.index(":")])
					rs=int(s[s.index(":")+1:])
					if x[4] == 'H':
						re+=0.2 if ls > rs else 0
					elif x[4] == 'A':
						re+=0.2 if rs > ls else 0
					else:
						pass
		return re

	def form(number):
		dict_teams={
			1: f"https://www.transfermarkt.de/hertha-bsc/vereinsspielplan/verein/44/plus/0?saison_id={str(num)}&heim_gast=",
			2: f"https://www.transfermarkt.de/eintracht-frankfurt/vereinsspielplan/verein/24/plus/0?saison_id={str(num)}&heim_gast=",
			3: f"https://www.transfermarkt.de/borussia-monchengladbach/vereinsspielplan/verein/18/plus/0?saison_id={str(num)}&heim_gast=",
			4: f"https://www.transfermarkt.de/1-fc-union-berlin/vereinsspielplan/verein/89/plus/0?saison_id={str(num)}&heim_gast=",
			5: f"https://www.transfermarkt.de/bayer-04-leverkusen/vereinsspielplan/verein/15/plus/0?saison_id={str(num)}&heim_gast=",
			6: f"https://www.transfermarkt.de/rasenballsport-leipzig/vereinsspielplan/verein/23826/plus/0?saison_id={str(num)}&heim_gast=",
			7: f"https://www.transfermarkt.de/1-fsv-mainz-05/vereinsspielplan/verein/39/plus/0?saison_id={str(num)}&heim_gast=",
			8: f"https://www.transfermarkt.de/vfb-stuttgart/vereinsspielplan/verein/79/plus/0?saison_id={str(num)}&heim_gast=",
			9: f"https://www.transfermarkt.de/fc-augsburg/vereinsspielplan/verein/167/plus/0?saison_id={str(num)}&heim_gast=",
			10: f"https://www.transfermarkt.de/borussia-dortmund/vereinsspielplan/verein/16/plus/0?saison_id={str(num)}&heim_gast=",
			11: f"https://www.transfermarkt.de/arminia-bielefeld/vereinsspielplan/verein/10/plus/0?saison_id={str(num)}&heim_gast=",
			12: f"https://www.transfermarkt.de/1-fc-koln/vereinsspielplan/verein/3/plus/0?saison_id={str(num)}&heim_gast=",
			13: f"https://www.transfermarkt.de/fc-schalke-04/vereinsspielplan/verein/33/plus/0?saison_id={str(num)}&heim_gast=",
			14: f"https://www.transfermarkt.de/sv-werder-bremen/vereinsspielplan/verein/86/plus/0?saison_id={str(num)}&heim_gast=",
			15: f"https://www.transfermarkt.de/tsg-1899-hoffenheim/vereinsspielplan/verein/533/plus/0?saison_id={str(num)}&heim_gast=",
			16: f"https://www.transfermarkt.de/fc-bayern-munchen/vereinsspielplan/verein/27/plus/0?saison_id={str(num)}&heim_gast=",
			17: f"https://www.transfermarkt.de/sc-freiburg/vereinsspielplan/verein/60/plus/0?saison_id={str(num)}&heim_gast=",
			18: f"https://www.transfermarkt.de/vfl-wolfsburg/vereinsspielplan/verein/82/plus/0?saison_id={str(num)}&heim_gast=",
			19: f"https://www.transfermarkt.de/fortuna-dusseldorf/vereinsspielplan/verein/38/plus/0?saison_id={str(num)}&heim_gast=",
			20: f"https://www.transfermarkt.de/sc-paderborn-07/vereinsspielplan/verein/127/plus/0?saison_id={str(num)}&heim_gast=",
			21: f"https://www.transfermarkt.de/hannover-96/vereinsspielplan/verein/42/plus/0?saison_id={str(num)}&heim_gast=",
			22: f"https://www.transfermarkt.de/1-fc-nurnberg/vereinsspielplan/verein/4/plus/0?saison_id={str(num)}&heim_gast=",
		}

		headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
		page = requests.get(dict_teams[number], headers=headers)
		soup = BeautifulSoup(page.content, 'html.parser')

		print(str(round((number/22)*100, 4))+" %")

		all=[]
		top = soup.find("div", {"class": "responsive-table"})
		table_body=top.find('tbody')
		rows = table_body.find_all('tr')
		for row in rows:
			cols=row.find_all('td')
			cols=[x.text.strip() for x in cols]
			if cols[1]=='':
				all.append(cols)

		return all

	print("Initializing teams games for this season:")
	init_forms() #get each teams plays from that season

	for i in range(1, max_matchday):

		# SCRAPING
		headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
		page = requests.get(f'https://www.transfermarkt.de/1-bundesliga/spieltag/wettbewerb/L1/plus/?saison_id={str(num)}&spieltag={str(curr_matchday)}', headers=headers)
		soup = BeautifulSoup(page.content, 'html.parser')
		curr_matchday += 1

		w,m,n,t,p,f=[],[],[],[],[],[]

		# PROCESSING
		teams_names = soup.findAll('td', class_='rechts hauptlink no-border-rechts hide-for-small spieltagsansicht-vereinsname')
		teams_names2 = soup.findAll('td', class_='hauptlink no-border-links no-border-rechts hide-for-small spieltagsansicht-vereinsname')
		scores = soup.findAll('td', class_='zentriert hauptlink no-border-rechts no-border-links spieltagsansicht-ergebnis')
		left=[]
		right=[]
		'''for i in teams_names:
			strt = i.text.strip().replace("	","").replace("\n","")
			left.append(strt)
		left = list(dict.fromkeys(left)) # remove duplicates from list left
		print(left)'''

		for i in teams_names:
			strt = i.text.strip().replace("	","").replace("\n","")
			left.append([strt[5:].strip(), re.sub('\D', '', strt[0:4])])
		for i in teams_names2:
			strt = i.text[:40].replace("	","").replace("\n","")
			right.append([strt[:-5].strip(), re.sub('\D', '', strt[-4:-1])])
		
		nt=[] # temporal list with names
		for i in range(0, len(left)):
			nt.append([replace_names(left[i][0]), replace_names(right[i][0])])
			p.append([int(left[i][1]), int(right[i][1])])

		for i in nt: n.append([teams[i[0]][0],teams[i[1]][0]]) #replace names with numbers
		for i in nt: w.append([teams[i[0]][1],teams[i[1]][1]]) #replace names with team worths

		for i in scores:
			raw_str = i.text.strip()
			end = 0
			ls=int(raw_str[:raw_str.index(":")])
			rs=int(raw_str[raw_str.index(":")+1:])
			end=1 if ls > rs else 0
			m.append(end)
		
		for i in n:
			a,b=0,0
			a+=spec_form(forms[i[0]-1])
			b+=spec_form(forms[i[1]-1])
			f.append([a, b])

		print(f"----- MATCHDAY {curr_matchday-1} DONE ----- ")# \n\nTeams: \n{n} \nScore: \n{m} \nPlace in Table: \n{p} \nWorth: \n{w} \nForm: \n{f} \nMatchday: \n{curr_matchday-1}")
		final_n.extend(n)
		final_m.extend(m)
		final_p.extend(p)
		final_w.extend(w)
		final_f.extend(f)
		for i in range(1,10):
			final_md.append(curr_matchday-1)

	final=[]
	for c,i in enumerate(final_n, 0):
		final.append([i[0], i[1], final_m[c], final_w[c][0], final_w[c][1], final_p[c][0], final_p[c][1], final_f[c][0], final_f[c][1], final_md[c]])
	df = pd.DataFrame(final, columns = ["home_team", "away_team", "home_team_win","ht_worth","at_worth","ht_table","at_table","ht_fs","aw_fs","curr_matchday"])
	df.to_csv('out_data_2019.csv')
	return df

#home_team, away_team, home_team_win [2,1,0], ht_full_worth, at_full_worth, ht_table, at_table, ht_fs, aw_fs, curr_matchday, 1stteamplayers_disabled, weather

del list[1]
list.remove()

if __name__ == '__main__':
	season(2019)