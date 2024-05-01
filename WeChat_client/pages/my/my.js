// pages/my/my.js
const app = getApp()
let interval1 = null;
Page({

  /**
   * 页面的初始数据
   */
  data: {
    text: {},
    username: '',
  },
  //退出登录
  eraseInfo(){
    app.globalData.username = "";
    app.globalData.status = "";
    app.globalData.token = "";
    wx.switchTab({
      url: '../login/login',
    })
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {

  },
  onShow() {

    var that = this
     
        
          that.setData({
          username:app.globalData.username,
          })
    if (typeof this.getTabBar === 'function' &&
      this.getTabBar()) {
      this.getTabBar().setData({
        selected: 1
      })
    }
  },


  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */


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