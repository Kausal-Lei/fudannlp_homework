#include<bits/stdc++.h>
using namespace std;
map<string,int> mp;
bool isdigit(string s) {
    for(auto i:s) {
        if(!isdigit(i)) return 0;
    }
    return 1;
}
string get(string s) {
    string t=s;
    for(int i=0;i<t.size();i++) if(t[i]>='A'&&t[i]<='Z') t[i]+='a'-'A';
    return t;
}
int main() {
    string s;
    freopen("mix.txt","r",stdin);
    while(cin>>s) {
        if(!isdigit(s)) mp[get(s)]=1;
    }
    printf("%d\n",mp.size());
    //for(auto i:mp) cout<<i.first<<endl;
}
