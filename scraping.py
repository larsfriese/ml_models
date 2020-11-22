import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

def s2021():

	#! CANT PROCESS SCORES THAT HAVE 2 DIGITS
	#! ONLY WORKS FOR SEASON 2020/2021

	max_matchday=5
	curr_matchday=1 #changes in loop

	dict_avg_worth_2020 = {
	"Hertha BSC": [1, 8.69],
	"Eintracht Frankfurt": [2, 6.16],
	"Borussia Mönchengladbach": [3, 10.11],
	"1. FC Union Berlin": [4, 2.08],
	"Bayer 04 Leverkusen": [5, 12.44],
	"RB Leipzig": [6, 17.15],
	"1. FSV Mainz 05": [7, 3.27],
	"VfB Stuttgart": [8, 2.42],
	"FC Augsburg": [9, 3.71],
	"Borussia Dortmund": [10, 21.75],
	"Arminia Bielefeld": [11, 1.68],
	"1. FC Köln": [12, 3.18],
	"FC Schalke 04": [13, 5.92],
	"Werder Bremen": [14, 3.94],
	"TSG Hoffenheim": [15, 7.47],
	"Bayern München": [16, 33.42],
	"SC Freiburg": [17, 3.93],
	"VfL Wolfsburg": [18, 7.47]
	}	

	def form(number):
		dict_teams={
			1: "https://www.betexplorer.com/soccer/team/hertha-berlin/2y0u8Wrq/results/",
			2: "https://www.betexplorer.com/soccer/team/eintracht-frankfurt/8vndvXTk/results/",
			3: "https://www.betexplorer.com/soccer/team/b-monchengladbach/88HSzjDr/results/",
			4: "https://www.betexplorer.com/soccer/team/union-berlin/pzHW4oaE/results/",
			5: "https://www.betexplorer.com/soccer/team/bayer-leverkusen/4jcj2zMd/results/",
			6: "https://www.betexplorer.com/soccer/team/rb-leipzig/KbS1suSm/results/",
			7: "https://www.betexplorer.com/soccer/team/mainz/EuakNmc1/results/",
			8: "https://www.betexplorer.com/soccer/team/vfb-stuttgart/nJQmYp1B/results/",
			9: "https://www.betexplorer.com/soccer/team/augsburg/fTVNku3I/results/",
			10: "https://www.betexplorer.com/soccer/team/dortmund/nP1i5US1/results/",
			11: "https://www.betexplorer.com/soccer/team/arminia-bielefeld/pp38UXK8/results/",
			12: "https://www.betexplorer.com/soccer/team/1-fc-koln/WG9pOTse/results/",
			13: "https://www.betexplorer.com/soccer/team/schalke/0Ija0Ej9/",
			14: "https://www.betexplorer.com/soccer/team/werder-bremen/Ig1f1fy3/results/",
			15: "https://www.betexplorer.com/soccer/team/hoffenheim/hQAtP9Sl/results/",
			16: "https://www.betexplorer.com/soccer/team/bayern-munich/nVp0wiqd/results/",
			17: "https://www.betexplorer.com/soccer/team/freiburg/fiEQZ7C7/results/",
			18: "https://www.betexplorer.com/soccer/team/wolfsburg/nwkTahLL/results/",
		}
		page = requests.get(dict_teams[number])
		soup = BeautifulSoup(page.content, 'html.parser')
		wls = soup.findAll("td",class_='table-main__formicon')
		wls2 = soup.findAll("td")
		for x in list(wls2):
			if str(curr_matchday)+". Round" == x.text.strip():
				t = list(wls2)[:list(wls2).index(x)]
				counter=1
				for i in t:
					if i.text.strip() == "details": counter +=1
				w=[]
				for i in list(wls)[counter:counter+5]:
					if "icon icon__w" in str(i):
						w.append(1) 
					elif "icon icon__d" in str(i):
						w.append(0.5)
					else:
						w.append(0)
				return w

	final_n,final_p,final_m,final_w,final_f=[],[],[],[],[]

	for i in range(1, max_matchday):

		# SCRAPING
		URL = 'https://www.dfb.de/bundesliga/spieltagtabelle/?spieledb_path=/competitions/12/season0s/current/matchday/1&spieledb_path=%2Fcompetitions%2F12%2Fseasons%2Fcurrent%2Fmatchday%2F' + str(curr_matchday) #change string end to 2 for second
		URLb = 'https://www.dfb.de/bundesliga/spieltagtabelle/?spieledb_path=/competitions/12/season0s/current/matchday/1&spieledb_path=%2Fcompetitions%2F12%2Fseasons%2Fcurrent%2Fmatchday%2F' + str(curr_matchday-1) if curr_matchday != 0 else 'https://www.dfb.de/bundesliga/spieltagtabelle/?spieledb_path=/competitions/12/season0s/current/matchday/1&spieledb_path=%2Fcompetitions%2F12%2Fseasons%2Fcurrent%2Fmatchday%2F' + str(curr_matchday)
		curr_matchday += 1
		page = requests.get(URL)
		page2 = requests.get(URLb)
		soup = BeautifulSoup(page.content, 'html.parser')
		soup2 = BeautifulSoup(page2.content, 'html.parser')

		# PROCESSING
		w,m,n,t,p,f=[],[],[],[],[],[]
		teams_names = soup.find_all("td", class_="column-team-emblem")
		teams_table = soup2.find_all("span", class_="hidden-xs")
		for i in teams_names:
			img = i.find('img', alt=True)
			n.append(img['alt'][12:])
		matches_results = soup.find_all("td", class_="column-score")

		for i in matches_results: m.append(i.text.strip())
		for i in teams_table: t.append(i.text.strip())
		t=t[2:20]
		for i in t: t[t.index(i)] = dict_avg_worth_2020[i][0] #change names to numbers

		for i in n: w.append(dict_avg_worth_2020[i][1])
		for i in n: n[n.index(i)] = dict_avg_worth_2020[i][0] #change names to numbers
		for i in m: #change scores to win for team 1 1/0
			if int(i[0])>int(i[-1]):
				m[m.index(i)] = 1 
			elif int(i[0])==int(i[-1]):
				m[m.index(i)] = 0.5 
			else:
				m[m.index(i)] = 0

		for i in n: p.append(t.index(i)+1)

		n_new=[]
		for c,i in enumerate(n, 0):
			if c % 2 == 0: n_new.append([n[n.index(i)], n[n.index(i)+1]])
		n=n_new

		p_new=[]
		for c,i in enumerate(p, 0):
			if c % 2 == 0: p_new.append([p[p.index(i)], p[p.index(i)+1]])
		p=p_new

		w_new=[]
		for c,i in enumerate(w, 0):
			if c % 2 == 0: w_new.append([w[w.index(i)], w[w.index(i)+1]])
		w=w_new

		for i in n:
			a,b=0,0
			for x in form(i[0]): a+=x
			for y in form(i[1]): b+=y
			f.append([a, b])

		#print(f"Teams: \n{n} \nScore: \n{m} \nPlace in Table: \n{p} \nWorth: \n{w} \nForm: \n{f}")
		final_n.extend(n)
		final_m.extend(m)
		final_p.extend(p)
		final_w.extend(w)
		final_f.extend(f)

	final=[]
	for c,i in enumerate(final_n, 0):
		final.append([i[0], i[1], final_m[c], final_w[c][0], final_w[c][1], final_p[c][0], final_p[c][1], final_f[c][0], final_f[c][1]])
	df = pd.DataFrame(final, columns = ["home_team", "away_team", "home_team_win","ht_worth","at_worth","ht_table","at_table","ht_fs","aw_fs"])
	df.to_csv('out_data.csv')
	return df

def season(num):
	max_matchday=34
	curr_matchday=1 #changes in loop

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
	"SC Paderborn": [20],
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
		else:
			name_result = name
		return name_result


	for c,i in enumerate(names_results):
		b = worths_results[c].text.strip()
		if 'Mio.' in b:
			b=float((b[:-7]).replace(',','.'))
		elif 'Tsd.' in b:
			b=float((b[:-7]/1000).replace(',','.'))
		else:
			pass
		teams[i.text.strip()].append(b)

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
		
		re = 0
		all=[]
		top = soup.find("div", {"class": "responsive-table"})
		table_body=top.find('tbody')
		rows = table_body.find_all('tr')
		for row in rows:
			cols=row.find_all('td')
			cols=[x.text.strip() for x in cols]
			if cols[1]=='':
				all.append(cols)
		
		for i in all:
			if str(curr_matchday)==i[2]:
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

	for i in range(1, max_matchday):

		# SCRAPING
		headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
		page = requests.get(f'https://www.transfermarkt.de/1-bundesliga/spieltag/wettbewerb/L1/plus/?saison_id={str(num)}&spieltag={str(curr_matchday)}', headers=headers)
		soup = BeautifulSoup(page.content, 'html.parser')
		curr_matchday += 1

		final_n,final_p,final_m,final_w,final_f=[],[],[],[],[]
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
		
		print([left, right])
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
			a+=form(i[0])
			b+=form(i[1])
			f.append([a, b])

		print(f"----- MATCHDAY {curr_matchday-1} DONE -----: ")# \n\nTeams: \n{n} \nScore: \n{m} \nPlace in Table: \n{p} \nWorth: \n{w} \nForm: \n{f} \nMatchday: \n{curr_matchday-1}")
		final_n.extend(n)
		final_m.extend(m)
		final_p.extend(p)
		final_w.extend(w)
		final_f.extend(f)

	final=[]
	for c,i in enumerate(final_n, 0):
		final.append([i[0], i[1], final_m[c], final_w[c][0], final_w[c][1], final_p[c][0], final_p[c][1], final_f[c][0], final_f[c][1], curr_matchday-1])
	df = pd.DataFrame(final, columns = ["home_team", "away_team", "home_team_win","ht_worth","at_worth","ht_table","at_table","ht_fs","aw_fs","curr_matchday"])
	df.to_csv('out_data.csv')
	return df

if __name__ == '__main__':
	#print(s2021())
	season(2018)