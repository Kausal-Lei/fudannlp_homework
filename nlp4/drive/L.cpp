#include<bits/stdc++.h>
#include <stdio.h>

using namespace std;
typedef unsigned long long ull;
bool IOerror=0;
inline char nc(){
    static char buf[100000],*p1=buf+100000,*pend=buf+100000;
    if (p1==pend){
        p1=buf; pend=buf+fread(buf,1,100000,stdin);
        if (pend==p1){IOerror=1;return -1;}
    }
    return *p1++;
}
bool blank(char ch){return ch==' '||ch=='\n'||ch=='\r'||ch=='\t';}
bool blank2(char ch){return ch=='\n'||ch=='\r'||ch=='\t';}
bool readline(char *s){
    char ch=nc();
    if (IOerror)return 0;
    for (;!blank2(ch)&&!IOerror;ch=nc())*s++=ch;
    *s=0;
    return 1;
}
void get(string a,vector<pair<string,string>> &v) {
    static char str[123123];
    strcpy(str,a.c_str());
    const char*split=" ";
    char *p;
    p=strtok(str,split);
    vector<string>tmp;
    while(p!=NULL) {
        tmp.push_back(p);
        p = strtok(NULL,split);
    }
    int ok=0;
    for(auto i:tmp[0]) if(isdigit(i)) ok=1;
    //if(ok&&tmp.back()=="O") v.push_back({"1000000007",tmp.back()});
    //else v.push_back({tmp[0],tmp.back()});
    v.push_back({tmp[0],tmp.back()});
}
int main() {
    char s[1230132];
	int ans=0;
    freopen("dev.txt","r",stdin);
    freopen("dev.in","w",stdout);
    vector<vector<pair<string,string>>>v;
    vector<pair<string,string>> tmp;
	while(readline(s)) {
        if(strlen(s)<=1) {
            v.push_back(tmp);
            tmp.clear();
            continue;
        }
        string a=s;
        get(a,tmp);
	}
	for(auto i:v) {
        for(auto j:i) cout<<j.first<<" ";cout<<endl;
        for(auto j:i) cout<<j.second<<" ";cout<<endl;
	}
}
