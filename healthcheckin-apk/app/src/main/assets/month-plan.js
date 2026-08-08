(function(){
const KP='health_month_plan_v21';
const monthPlan=[
['启动日','固定今晚睡觉和明早起床时间','轻快走20分钟','记录今天烟和槟榔真实数量','睡前完成2分钟复盘'],
['稳住早餐','早餐加入鸡蛋/牛奶/豆制品任选一项','累计步行25分钟','今天不喝含糖饮料','睡前按提醒开始收尾'],
['饭后动起来','午饭后走10分钟','晚饭后走10分钟','吃1份水果或蔬菜','记录最容易想抽烟/嚼槟榔的时刻'],
['肩颈放松日','累计步行30分钟','做5分钟肩颈和髋部拉伸','睡前30分钟停止工作型内容','按时完成复盘'],
['减少自动化习惯','累计步行30分钟','挑1次“其实不太想抽”的烟不点','含糖饮料保持0','睡够目标时长'],
['饮食结构日','累计步行35分钟','至少两餐有明确蛋白质','给自己安排一个不嚼槟榔的连续时段','晚上不过量夜宵'],
['第1周复盘','轻松走25分钟','晨起称一次体重','复盘这一周最影响精力的因素','今天只恢复，不加码训练'],
['力量入门A','椅子起立/深蹲 3×8','斜板俯卧撑 3×8','臀桥 3×12','再轻松走20分钟'],
['恢复走','快走35分钟','吃1份水果或蔬菜','今天不喝含糖饮料','按睡前提醒收尾'],
['烟触发点替代','累计步行30分钟','选1个固定抽烟场景改成喝水/走3分钟','记录烟和槟榔总量','睡前复盘替代是否有效'],
['力量入门B','椅子起立/深蹲 3×10','背包划船 3×10','平板支撑 2×20秒','轻走15分钟放松'],
['心肺建立','快走40分钟','至少两餐有蛋白质','夜宵尽量取消','睡眠目标优先'],
['低压恢复','轻松走30分钟','拉伸8分钟','今天不追求强度','早点进入睡前模式'],
['第2周复盘','晨起称体重','轻松走25分钟','比较近7天平均睡眠和第1周','给下周写1个最重要调整'],
['进入稳定期','快走35分钟','睡眠达到目标','完成每日复盘','记录今日精力评分'],
['力量A进阶','深蹲 3×10-12','斜板俯卧撑 3×10','臀桥 3×15','平板支撑 2×25秒'],
['活动巩固','累计活动30分钟','饭后至少走一次10分钟','含糖饮料0','今天保持轻松'],
['耐力日','快走40分钟','其中10分钟稍快速度','两餐有蛋白质','按时睡觉'],
['力量B进阶','深蹲 3×12','背包划船 3×12','臀桥 3×15','平板支撑 2×30秒'],
['长一点的步行','累计步行45分钟','可以拆成2-3段完成','记录今天最低精力出现的时间','不要因为运动额外熬夜'],
['第3周复盘','轻松走25分钟','晨起称体重','统计这一周烟/槟榔是否下降','如果准备好了，考虑确定正式戒烟日期'],
['第四周启动','快走40分钟','睡眠目标优先','含糖饮料0','完成复盘'],
['力量A稳定','深蹲 3×12','斜板俯卧撑 3×10-12','臀桥 3×15','平板支撑 2×30秒'],
['恢复与饮食','累计活动30分钟','至少1份蔬菜+1份水果','两餐有蛋白质','今晚不额外加练'],
['心肺稳定','快走40分钟','其中15分钟稍快','记录今日烟/槟榔数量','按时进入睡眠流程'],
['力量B稳定','深蹲 3×12','背包划船 3×12-15','臀桥 3×15','平板支撑 2×35秒'],
['本月最长步行','累计步行45分钟','可拆成午后20+晚上25分钟','今天不喝含糖饮料','睡前复盘身体感觉'],
['第4周复盘','轻松走25分钟','晨起称体重','比较第1周和第4周精力','找出最值得长期保留的3个习惯'],
['综合执行日','活动30-40分钟','按计划吃三餐且避免暴食','尽量减少烟/槟榔触发行为','睡眠+复盘全部完成'],
['30天总结日','晨起称体重','轻松走30分钟','总结30天最明显的3个变化','为下一个30天只保留3个核心目标']
];
function dayIndex(){try{const s=settings().startDate||dstr();return Math.min(30,Math.max(1,days(s,dstr())+1));}catch(e){return 1}}
function loadDone(){try{return JSON.parse(localStorage.getItem(KP)||'{}')}catch(e){return{}}}
function saveDone(x){localStorage.setItem(KP,JSON.stringify(x))}
function phaseName(d){if(d<=7)return'第1阶段 · 稳住节奏';if(d<=14)return'第2阶段 · 加入力量';if(d<=21)return'第3阶段 · 提升体能';return'第4阶段 · 固化习惯'}
function phaseColor(d){if(d<=7)return'#dff2e5';if(d<=14)return'#e7f0ff';if(d<=21)return'#fff1d9';return'#efe7ff'}
function injectStyles(){const s=document.createElement('style');s.textContent=`
.monthCard{background:linear-gradient(145deg,#fff,#f7fbf8);border:1px solid #dfeae3;border-radius:22px;padding:17px;margin:11px 0;box-shadow:0 4px 18px rgba(33,67,47,.04)}
.monthHead{display:flex;justify-content:space-between;gap:12px;align-items:center}.dayBubble{min-width:66px;height:66px;border-radius:19px;background:#315f49;color:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center}.dayBubble b{font-size:22px}.dayBubble span{font-size:10px;opacity:.85}.monthTitle{font-size:18px;font-weight:850}.monthPhase{display:inline-block;margin-top:5px;padding:5px 9px;border-radius:999px;font-size:11px;font-weight:750;color:#315f49}.taskLine{display:flex;gap:10px;align-items:flex-start;padding:10px 0;border-bottom:1px dashed #e4ebe6}.taskLine:last-child{border-bottom:0}.taskLine input{width:20px;height:20px;margin-top:1px;accent-color:#315f49}.taskLine.done span{text-decoration:line-through;color:#9ba59f}.monthProgress{height:10px;border-radius:99px;background:#e7eee9;overflow:hidden;margin:12px 0}.monthProgress>div{height:100%;background:linear-gradient(90deg,#315f49,#65a37f)}.planDay{padding:14px;border-radius:17px;border:1px solid #e5ebe7;background:#fff;margin:10px 0}.planDay.current{border:2px solid #5b9373;background:#f4faf6}.planDayTop{display:flex;justify-content:space-between;align-items:center}.planDay b{font-size:15px}.planDay small{color:#7a867f}.planTasks{margin-top:8px;font-size:13px;color:#647068;line-height:1.7}.monthHero{background:linear-gradient(145deg,#edf8f1,#fff);border:1px solid #dbeade;border-radius:22px;padding:18px;margin:10px 0}.monthHero h2{margin:5px 0 8px;font-size:23px}.monthStatGrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.monthStat{background:white;border:1px solid #e2ebe5;border-radius:16px;padding:12px}.monthStat b{font-size:24px;display:block}.monthStat span{font-size:11px;color:#78847c}@media(max-width:390px){.monthHead{align-items:flex-start}}
`;document.head.appendChild(s)}
function doneCountForDay(d){const x=loadDone()[d]||{};return Object.values(x).filter(Boolean).length}
function totalDone(){const x=loadDone();let c=0;for(let i=1;i<=30;i++)c+=Object.values(x[i]||{}).filter(Boolean).length;return c}
function renderTodayPlan(){const d=dayIndex(),p=monthPlan[d-1],done=loadDone()[d]||{};let host=document.getElementById('monthTodayCard');if(!host)return;let count=doneCountForDay(d),pct=Math.round(count/4*100);host.innerHTML=`<div class="monthHead"><div><div class="eyebrow">30 DAY ACTION PLAN</div><div class="monthTitle">${p[0]}</div><span class="monthPhase" style="background:${phaseColor(d)}">${phaseName(d)}</span></div><div class="dayBubble"><b>${d}</b><span>DAY / 30</span></div></div><div class="monthProgress"><div style="width:${pct}%"></div></div><div class="muted">今日任务完成 ${count}/4</div><div id="todayMonthTasks"></div><button class="btn soft" id="openMonthPlan">查看完整30天计划</button>`;let box=host.querySelector('#todayMonthTasks');p.slice(1).forEach((t,i)=>{let row=document.createElement('label');row.className='taskLine'+(done[i]?' done':'');row.innerHTML=`<input type="checkbox" ${done[i]?'checked':''}><span>${t}</span>`;row.querySelector('input').onchange=e=>{let x=loadDone();x[d]=x[d]||{};x[d][i]=e.target.checked;saveDone(x);renderTodayPlan();renderMonthPage()};box.appendChild(row)});host.querySelector('#openMonthPlan').onclick=()=>{const btn=document.querySelector('.nav button[data-month-plan]');showMonthPage(btn)}}
function renderMonthPage(){let page=document.getElementById('monthplan');if(!page)return;let d=dayIndex(),current=monthPlan[d-1],td=totalDone(),all=120,pct=Math.round(td/all*100);page.innerHTML=`<div class="monthHero"><div class="eyebrow">YOUR FIRST 30 DAYS</div><h2>第 ${d} 天 · ${current[0]}</h2><div class="desc">${phaseName(d)}。这个月不追求猛练，目标是把睡眠、活动、力量和习惯管理变成能长期维持的节奏。</div><div class="monthProgress"><div style="width:${pct}%"></div></div><div class="monthStatGrid"><div class="monthStat"><b>${pct}%</b><span>30天任务总体完成</span></div><div class="monthStat"><b>${td}</b><span>已完成任务 / 120</span></div></div></div><div id="monthDays"></div>`;let list=page.querySelector('#monthDays');monthPlan.forEach((p,i)=>{let day=i+1,dc=doneCountForDay(day),el=document.createElement('div');el.className='planDay'+(day===d?' current':'');el.innerHTML=`<div class="planDayTop"><b>Day ${day} · ${p[0]}</b><small>${dc}/4</small></div><div class="planTasks">${p.slice(1).map(x=>'• '+x).join('<br>')}</div>`;list.appendChild(el)})}
function showMonthPage(btn){document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));document.getElementById('monthplan').classList.add('active');document.querySelectorAll('.nav button').forEach(x=>x.classList.remove('active'));if(btn)btn.classList.add('active');renderMonthPage();scrollTo(0,0)}
function inject(){injectStyles();let today=document.getElementById('today');if(today){let card=document.createElement('div');card.id='monthTodayCard';card.className='monthCard';let hero=today.querySelector('.hero');if(hero&&hero.nextSibling)today.insertBefore(card,hero.nextSibling);else today.prepend(card)}let app=document.querySelector('.app');let page=document.createElement('section');page.id='monthplan';page.className='page';app.appendChild(page);let nav=document.querySelector('.navin');if(nav){nav.style.gridTemplateColumns='repeat(6,1fr)';let b=document.createElement('button');b.setAttribute('data-month-plan','1');b.innerHTML='<i>☷</i>计划';b.onclick=()=>showMonthPage(b);nav.insertBefore(b,nav.children[1]);nav.querySelectorAll('button').forEach(x=>x.style.fontSize='10.5px')}renderTodayPlan();renderMonthPage()}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',inject);else inject();
})();
