var app = getApp()
// let status = 'administrator';  //接收到数据注释掉
Component({
  data: {
    selected: 0,
    color: "#000000",
    roleId: '',
    selectedColor: "#1396DB",
    allList: [{
      list1: [{

        "pagePath": "/pages/dist/dist",
        "text": "列表",
        "iconPath": "/images/dist-tab.png",
        "selectedIconPath": "/images/dist-tab.png"

      },
      {
        "pagePath": "/pages/my/my",
        "text": "个人",
        "iconPath": "/images/user-tab.png",
        "selectedIconPath": "/images/user-tab.png"

      }]
    }],
    list: []
  },
  attached() {
    const roleId = wx.getStorageSync('status')//app.globalData.status获取的数据
    //app.globalData.status
    if (app.globalData.status == 'user') {
      this.setData({
        list: this.data.allList[0].list1
      })
    }
  },
  methods: {
    switchTab(e) {
      const data = e.currentTarget.dataset
      const url = data.path
      this.setData({
        selected: data.path
      })
      wx.switchTab({
        url,
        // success: function (e) {
        //   var page = getCurrentPages().pop();
        //   if (page == undefined || page == null) return;
        //   page.onLoad();
        // }
      })
      
    }
  },



})


