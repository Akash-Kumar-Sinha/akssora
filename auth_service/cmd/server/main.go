package main

import (
	"auth_service/api"
	"auth_service/internals/database"
	"auth_service/internals/helpers"
	"os"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

func init() {
	godotenv.Load()
	database.LoadInitializers()
	database.ConnectToDb()
	helpers.InitKeys()
}

func main() {
	router := gin.Default()
	PORT := os.Getenv("PORT")

	if PORT == "" {
		PORT = "8000"
	}

	frontendDomain := os.Getenv("FRONTEND_DOMAIN")

	router.Use(cors.New(cors.Config{
		AllowOrigins:     []string{frontendDomain},
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Authorization"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))
	router.GET("/", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "You are under my protection",
		})
	})

	router.GET("/api",
		func(c *gin.Context) {
			c.Redirect(301, "/")
		})
	router.GET("/api/v1", func(c *gin.Context) {
		c.Redirect(301, "/")
	})

	groupapi := router.Group("/api/v1")

	auth := groupapi.Group("/auth")
	api.AuthRoutes(auth)

	app := groupapi.Group("/app")
	api.AppRoutes(app)

	router.Run("0.0.0.0:" + PORT)

}
