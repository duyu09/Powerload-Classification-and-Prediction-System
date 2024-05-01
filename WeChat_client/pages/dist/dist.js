// pages/liebiao/liebiao.js
let interval1=0;
const app = getApp()
Page({

  /**
   * 页面的初始数据
   */
  data: {
    indexmenu:[{url:'dadad',id:'dada',text:'电力负载分类分解与预测',icon:'../../images/clothes_wash.png'}]
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {
    
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {
    
      
  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

    
//     var that =this
//     interval1=setInterval(function(){
//     wx.request({
//     // 10.106.17.36:2000/api/distAppliance127.0.0.1:5000
//     url: app.globalData.url+'/api/distAppliance',
//     data:{
//       userid:app.globalData.userid
//     },
//     method:'POST',
//     header: {
//     },
//     success:function(res){
//     that.setData({
//                   indexmenu:res.data,
//                 userid:app.globalData.userid})
//     console.log(that.data.indexmenu);
//     wx.setNavigationBarTitle({
//       title: app.globalData.village+app.globalData.housenumber
//   })
//   }
// })
//         },1000);

},

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {
    clearInterval(interval1);
  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})