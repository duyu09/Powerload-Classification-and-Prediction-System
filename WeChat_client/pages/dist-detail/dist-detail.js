import * as echarts from '../../ec-canvas/echarts.min';
 var app = getApp();
 let interval1=0;
  function getOption(){
    var x2 = []
    for(let i=1;i<=50;i++)
    {
      x2.push(i)
    }
   return{
       title: {
           text: '实时功率曲线 单位：瓦(W)',
           padding: [10, 0, 0, 20],
           textStyle: {
               fontSize: 14,
               color: '#696969'
           },
           top: '10rpx'
       },
       backgroundColor: "#fff",
       color: ["#006EFF", "#67E0E3", "#9FE6B8"],
       animation: false,
       grid: {
           show: false
       },
       xAxis: {
           type: 'category',
           data: x2,      
       },
       yAxis: {
           scale: true,
           type: 'value',
       },
       series: [{
         data:app.globalData.real_data,
           type: 'line',
           lineStyle: {
               width: 1
           },
           symbol:'line'
       }]
   };
 }

 function getOptionTwo(){
   var x1 = []
   for(let i=1;i<=20;i++)
   {
     x1.push(i)
   }
  return{
      title: {
          text: '预测功率曲线 单位：瓦(W)',
          padding: [10, 0, 0, 20],
          textStyle: {
              fontSize: 14,
              color: '#696969'
          },
          top: '10rpx'
      },
      backgroundColor: "#fff",
      color: ["red"],
      animation: false,
      grid: {
          show: false
      },
      xAxis: {
          type: 'category',
          data: x1,      
      },
      yAxis: {
          scale: true,
          type: 'value',
          // axisLabel: {
          //     formatter: function (value) {
          //     var val = value / 1000000000 + 'G';
          //         return val
          //     }
          // }
      },
      series: [{
        data:app.globalData.prediction,
          type: 'line',
          lineStyle: {
              width: 1
          },
          symbol:'line'
      }]
  };
}
 Page({
 
   /**
    * 页面的初始数据
    */
   data: {
     userid:"",
     title:"",
     label:"",
     energy:0,
     real_data:[],
    prediction:[],
    time:"",
    interval1:null,
    interval2:null,
     ecOne: {
         lazyLoad: true
     },
     ecTwo: {
         lazyLoad: true
     },
     
   text:{},
   ss:'',
       winHeight:"",//窗口高度
       currentTab:0, //预设当前项的值
       scrollLeft:0, //tab标题的滚动条位置
 
 },
  switchTab:function(e){
     this.setData({
       currentTab:e.detail.current
     });
     this.checkCor();
   },
   // 点击标题切换当前页时改变样式
   swichNav:function(e){
     var cur=e.target.dataset.current;
     if(this.data.currentTaB==cur){return false;}
     else{
       this.setData({
         currentTab:cur
       })
     }
   },
   checkCor:function(){
    if (this.data.currentTab>4){
     this.setData({
      scrollLeft:300
     })
    }else{
     this.setData({
      scrollLeft:0
     })
    }
   },
 
   /**
    * 生命周期函数--监听页面加载
    */
   onLoad(options) {
     var that=this;
     console.log("options",options)
         this.setData({
          title:options.title,
          userid:options.userid
         })
      
     this.oneComponent = this.selectComponent('#mychart-one')
     this.twoComponent = this.selectComponent('#mychart-two')
     setTimeout(()=>this.initone(),1000)
     setTimeout(()=>this.inittwo(),1000)
	 wx.getSystemInfo( { 
	       success: function( res ) { 
	         var clientHeight=res.windowHeight,
	           clientWidth=res.windowWidth,
	           rpxR=750/clientWidth;
	        var calc=clientHeight*rpxR-180;
	         console.log(calc)
	         that.setData( { 
	           winHeight: calc 
	         }); 
	       } 
	     });
   },
   initone(){
     this.oneComponent.init((canvas,width,height,dpr)=>{
       let chart=echarts.init(canvas,null,{
         width:width,
         height:height,
         devicePixelRatio:dpr
       })
 
       let option=getOption()
       chart.setOption(option)
       this.chart=chart
       return chart
     })
   },
   inittwo(){
     this.twoComponent.init((canvas,width,height,dpr)=>{
       let secondchart=echarts.init(canvas,null,{
         width:width,
         height:height,
         devicePixelRatio:dpr
       })
 
       let option=getOptionTwo()
       secondchart.setOption(option)
       this.secondchart=secondchart
       return secondchart
     })
   },
     footerTap:app.footerTap,
 
     onShow() {
       console.log(app.globalData.username,app.globalData.token)
      var that =this
      // console.log("test:",that.data.dataitem[that.data.atype])
      wx.request({
        method:'POST',
        data:{
          phone_number:app.globalData.username,
          token:app.globalData.token
        },
        url:  app.globalData.url+'/api/getdata',
        success:function(res){
          // var dataitems=res.data[0]
          console.log(res.data)
          // console.log(that.data.view)
          // console.log(dataitems[that.data.view])
            app.globalData.apptype=res.data.label
            app.globalData.real_data=res.data.real_time
            app.globalData.prediction=res.data.prediction
            // that.data.real_data=res.data.real_time
            // that.data.prediction = res.data.prediction
            const array = app.globalData.real_data;
            const sum = array.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
            // const energy = (sum * 60 / 1000).toFixed(2);
            const energy = Math.round(sum * 60 / 1000);
            
        // console.log(res.data)
        that.setData({
                      label:res.data.label,
                      real_data:res.data.real_time,
                      prediction:res.data.prediction,
                      energy: energy
                  }
                )
      }
    })
    that.initone();
    that.inittwo();

      that.data.interval1=setInterval(function(){
      wx.request({
      method:'POST',
      data:{
        phone_number:app.globalData.username,
        token:app.globalData.token
      },
      url:  app.globalData.url+'/api/getdata',
      // url: 'http://10.106.17.36:2000/api/distRetail',127.0.0.1:5000
      success:function(res){
        // var dataitems=res.data[0]
        console.log(res.data)
        // console.log(that.data.view)
        // console.log(dataitems[that.data.view])
          app.globalData.apptype=res.data.label
          app.globalData.real_data=res.data.real_time
          app.globalData.prediction=res.data.prediction
          // that.data.real_data=res.data.real_time
          // that.data.prediction = res.data.prediction

      // console.log(res.data)
      that.setData({
                    label:res.data.label,
                    real_data:res.data.real_time,
                    prediction:res.data.prediction
                }
              )
    }
  })
  that.initone();
  that.inittwo();
          },45000);


  that.data.interval2=setInterval(function(){
    const now = new Date();

    const year = now.getFullYear();
    const month = ('0' + (now.getMonth() + 1)).slice(-2);
    const day = ('0' + now.getDate()).slice(-2);
    const hours = ('0' + now.getHours()).slice(-2);
    const minutes = ('0' + now.getMinutes()).slice(-2);
    const seconds = ('0' + now.getSeconds()).slice(-2);
    
    const formattedTime = year + '年' + month + '月' + day +'日 '+ hours +':'+ minutes +':'+ seconds;
    
    
    
    that.setData({
      time:formattedTime
    })

  },1000)
  },
  onUnload() {
    clearInterval(this.data.interval1);
    clearInterval(this.data.interval2);
  }
 })